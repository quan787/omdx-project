# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
import numpy as np
from PIL import Image, ImageTk
from astropy.time import Time
import json
import os
from .observation import MeteorObservation
from importlib import resources


def get_resource_path(package_data, resource_name):
    return str(resources.files(package_data).joinpath(resource_name))


# 尝试引入 OpenCV 用于 Debayer
try:
    import cv2

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: OpenCV not found. Bayer pattern demosaicing will be disabled.")


class ImageProcessor:
    """
    负责图像的数值处理，如色阶调整、边缘提取等，与GUI分离。
    """

    @staticmethod
    def debayer_and_color_correct(image_data, color_mode):
        """
        处理色彩模式：Debayer 或 BGR/RGB 转换。
        尽量保持原始数据类型 (uint16/uint8) 以保留动态范围。
        """
        if image_data is None:
            return None

        # 没有任何处理需要的模式
        if color_mode is None or color_mode == "A":
            return image_data

        # 处理 Bayer 模式
        # OpenCV 的 Bayer 转换码映射
        bayer_map = {
            "RGGB": cv2.COLOR_BayerRG2RGB if HAS_OPENCV else None,
            "BGGR": cv2.COLOR_BayerBG2RGB if HAS_OPENCV else None,
            "GRBG": cv2.COLOR_BayerGR2RGB if HAS_OPENCV else None,
            "GBRG": cv2.COLOR_BayerGB2RGB if HAS_OPENCV else None,
        }

        if color_mode in bayer_map:
            if HAS_OPENCV:
                # OpenCV 通常支持 uint8 和 uint16。对于 float32 支持可能有限，
                # 如果是 float，这里暂时不做 debayer 或者先转 uint 可能会丢失精度。
                # 假设 MeteorObservation 的 bayer 数据通常是 uint。
                try:
                    # 注意：OpenCV 的 debayer 输出通常是 RGB (如果选了 *2RGB)
                    return cv2.cvtColor(image_data, bayer_map[color_mode])
                except Exception:
                    # 如果出错（例如数据类型不支持），返回原图
                    return image_data
            else:
                return image_data  # 无 OpenCV，直接返回马赛克原图

        # 处理 BGR 模式 (常见于 OpenCV 输出的数据)
        if color_mode == "BGR":
            # 转换为 RGB: 翻转最后一个维度
            if image_data.ndim == 3 and image_data.shape[2] == 3:
                return image_data[..., ::-1]

        return image_data

    @staticmethod
    def apply_levels(image_data, vmin, vmax, gamma=1.0):
        """
        将原始数据映射到 0-255 的 8bit 图像。
        兼容单通道(H,W) 和 三通道(H,W,3)。
        公式: out = ((in - vmin) / (vmax - vmin)) ^ (1/gamma) * 255
        """
        if image_data is None:
            return None

        # 避免除以零
        if vmax <= vmin:
            vmax = vmin + 1e-5

        data = image_data.astype(np.float32)
        data = (data - vmin) / (vmax - vmin)
        np.clip(data, 0, 1, out=data)

        if gamma != 1.0:
            np.power(data, 1.0 / gamma, out=data)

        return (data * 255).astype(np.uint8)

    @staticmethod
    def get_edges(mask):
        """
        简单的边缘提取，用于 signal_frame
        """
        if mask is None:
            return None
        # 使用 numpy 梯度计算边缘
        # 这是一个非常轻量的实现：如果在x或y方向上有变化，就是边缘
        gy, gx = np.gradient(mask.astype(float))
        edges = (gx**2 + gy**2) > 0
        return edges


class HistogramWindow(tk.Toplevel):
    """
    直方图与色阶调整窗口
    """

    def __init__(self, parent, update_callback):
        super().__init__(parent)
        self.title("直方图与色阶")
        self.geometry("500x400")
        self.update_callback = update_callback
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # 状态
        self.vmin = 0.0
        self.vmax = 255.0  # 默认值，会被 set_range 覆盖
        self.gamma = 1.0
        self.data_ref = None  # 当前图像数据的引用

        # 滑块的最大上限，根据数据类型动态调整
        self.slider_limit_max = 255.0

        # 布局 - 使用原生 Canvas 替代 Matplotlib
        self.canvas_height = 250
        self.canvas = tk.Canvas(self, bg="white", height=self.canvas_height)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # 绑定 resize 事件以重绘
        self.canvas.bind("<Configure>", self.on_resize_canvas)

        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # 定义变量用于绑定 Entry
        self.var_min = tk.DoubleVar(value=0.0)
        self.var_gamma = tk.DoubleVar(value=1.0)
        self.var_max = tk.DoubleVar(value=255.0)

        # 为每个变量添加监听，当模式为 "w" (写入/修改) 时调用 on_change
        self.var_min.trace_add("write", lambda *args: self.on_change())
        self.var_gamma.trace_add("write", lambda *args: self.on_change())
        self.var_max.trace_add("write", lambda *args: self.on_change())

        # 辅助函数：创建带输入框的滑动条行
        # 注意：Scale 的 to 参数现在需要动态更新，初始化时先给一个默认值
        def create_slider_row(row, label, var, from_, to_):
            ttk.Label(ctrl_frame, text=label).grid(row=row, column=0, padx=5)
            # 滑块
            s = ttk.Scale(
                ctrl_frame,
                from_=from_,
                to=to_,
                variable=var,
                command=lambda _: self.on_change(),
            )
            s.grid(row=row, column=1, sticky="ew")
            # 输入框
            e = ttk.Entry(ctrl_frame, textvariable=var, width=8)
            e.grid(row=row, column=2, padx=5)
            return s

        self.s_min = create_slider_row(0, "Min (黑场)", self.var_min, 0, 255)
        self.s_gamma = create_slider_row(1, "Gamma", self.var_gamma, 0.1, 5.0)
        self.s_max = create_slider_row(2, "Max (白场)", self.var_max, 0, 255)

        self.s_min.configure(command=self.on_change)
        self.s_gamma.configure(command=self.on_change)
        self.s_max.configure(command=self.on_change)

        ctrl_frame.columnconfigure(1, weight=1)
        self.withdraw()  # 初始隐藏

    def update_range_limits(self, data_type, max_val_ref=None):
        """
        根据数据类型更新滑块的上限
        """
        if issubclass(data_type, np.integer):
            if np.iinfo(data_type).bits > 8:
                self.slider_limit_max = 65535.0
            else:
                self.slider_limit_max = 255.0
        elif issubclass(data_type, np.floating):
            # 浮点数，如果有 max_val_ref (max_frame的最大值)，则使用它，否则默认 1.0 或尝试推断
            if max_val_ref is not None:
                self.slider_limit_max = float(max_val_ref)
                if self.slider_limit_max == 0:
                    self.slider_limit_max = 1.0
            else:
                self.slider_limit_max = 1.0

        # 更新滑块范围
        self.s_min.configure(to=self.slider_limit_max)
        self.s_max.configure(to=self.slider_limit_max)

        # 更新当前值的默认状态 (如果这是刚加载文件)
        self.vmax = self.slider_limit_max
        self.var_max.set(self.vmax)
        self.vmin = 0.0
        self.var_min.set(0.0)

    def on_change(self, _=None):
        try:
            # 只有当变量能正确解析为数字时才更新
            self.vmin = self.var_min.get()
            self.vmax = self.var_max.get()
            self.gamma = self.var_gamma.get()
            self.update_callback(self.vmin, self.vmax, self.gamma)
        except tk.TclError:
            pass

    def set_data(self, data):
        """接收图像数据并绘制直方图"""
        if data is None:
            return

        self.data_ref = data
        # 触发一次重绘
        self.draw_histogram()

    def on_resize_canvas(self, event):
        """窗口大小改变时重绘"""
        if self.data_ref is not None:
            self.draw_histogram()

    def draw_histogram(self):
        """使用 Canvas 手动绘制直方图"""
        if self.data_ref is None:
            return

        # 获取 Canvas 当前尺寸
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return

        flat = self.data_ref.flatten()
        # 简单下采样以提高直方图计算速度
        if len(flat) > 100000:
            flat = np.random.choice(flat, 100000)

        # 动态范围可能很大，限制一下 bins 数量
        bins_count = 64
        counts, bin_edges = np.histogram(flat, bins=bins_count)

        # Log 缩放处理
        counts_log = np.log1p(counts)
        max_count = np.max(counts_log)
        if max_count == 0:
            max_count = 1

        self.canvas.delete("all")

        # 预留底部边距用于显示文字
        margin_bottom = 20
        graph_h = h - margin_bottom

        # 绘制背景标题
        self.canvas.create_text(w / 2, 10, text="直方图 (Log Scale)", fill="#333")

        # 绘制条形图
        bar_w = w / bins_count
        for i in range(bins_count):
            val = counts_log[i]
            # 高度归一化，基于 graph_h 计算
            bar_h = (val / max_count) * (graph_h - 20)

            x0 = i * bar_w
            y0 = graph_h - bar_h
            x1 = (i + 1) * bar_w
            y1 = graph_h  # 底部对齐到 graph_h

            self.canvas.create_rectangle(x0, y0, x1, y1, fill="#888888", outline="")

        # === 新增：绘制底部 X 轴范围标注 ===
        min_val = bin_edges[0]
        max_val = bin_edges[-1]

        # 左下角显示最小值
        self.canvas.create_text(
            2, h, anchor="sw", text=f"{min_val:.1f}", fill="black", font=("Arial", 9)
        )
        # 右下角显示最大值
        self.canvas.create_text(
            w - 2,
            h,
            anchor="se",
            text=f"{max_val:.1f}",
            fill="black",
            font=("Arial", 9),
        )

    def on_close(self):
        self.withdraw()


class InfoWindow(tk.Toplevel):
    """通用表格展示窗口"""

    def __init__(self, parent, title, columns, data):
        super().__init__(parent)
        self.title(title)

        # 修改点1：根据列数动态计算初始宽度（每列100px），设置最小800px，最大1200px
        target_width = len(columns) * 100 + 20
        win_width = min(1200, max(800, target_width))
        self.geometry(f"{win_width}x400")

        # 修改点2：增加水平滚动条容器支持
        tree = ttk.Treeview(self, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor="center")

        # 垂直滚动条
        vsb = ttk.Scrollbar(self, orient="vertical", command=tree.yview)
        # 新增：水平滚动条
        hsb = ttk.Scrollbar(self, orient="horizontal", command=tree.xview)

        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # 布局：先放置滚动条，再放置 Treeview 填满剩余空间
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(side="left", fill="both", expand=True)

        for row in data:
            tree.insert("", "end", values=row)


class MeteorPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Meteor Observation Viewer")
        self.root.geometry("1024x768")

        # 字体设置
        self.font_family = self._get_chinese_font()
        self.custom_font = tkfont.Font(family=self.font_family, size=10)

        # 数据模型
        self.obs = None
        self.current_frame_idx = 0
        self.is_playing = False
        self.play_speed = 1.0  # 1.0 = 实时(假设25fps或根据曝光时间), 这里简化为固定延迟
        self.fps_delay = 40  # ms (25fps)

        # 视图控制
        self.zoom_scale = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.drag_start = None

        # 图像缓存
        self.img_tk = None  # 保持引用防止GC

        self.star_common_names = {}
        self.load_star_common_names()

        # 辅助窗口
        self.hist_window = HistogramWindow(self.root, self.update_image_display)

        self.setup_ui()

    def _get_chinese_font(self):
        """获取系统中最合适的无衬线中文字体"""
        import tkinter.font as tkfont

        all_fonts = tkfont.families()
        # 备选列表：思源、微软雅黑、苹方、文泉驿
        candidates = [
            "Source Han Sans CN",
            "Microsoft YaHei",
            "PingFang SC",
            "WenQuanYi Micro Hei",
            "sans-serif",
        ]
        for f in candidates:
            if f in all_fonts:
                return f
        return "sans-serif"  # 兜底

    def load_star_common_names(self):
        """从 JSON 加载恒星常用名对照表"""
        path = get_resource_path("omdx.data", "star_common_names.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 优先级：中文覆盖英文。如果 JSON 里有中文名，优先显示中文。
                    en_names = data.get("English", {})
                    cn_names = data.get("Chinese", {})
                    self.star_common_names = {**en_names, **cn_names}
            except Exception as e:
                print(f"Warning: 无法加载恒星名称表: {e}")

    def setup_ui(self):
        # === 区域1: 顶部工具栏 ===
        top_frame = ttk.Frame(self.root, padding=5)
        top_frame.pack(fill=tk.X)

        ttk.Button(top_frame, text="打开 FITS", command=self.open_fits).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(top_frame, text="保存 FITS", command=self.save_fits).pack(
            side=tk.LEFT, padx=5
        )

        # 内容标签容器
        self.contents_frame = ttk.Frame(top_frame)
        self.contents_frame.pack(side=tk.LEFT, padx=20)

        # === 区域2: 信息展示与按钮 ===
        info_frame = ttk.Frame(self.root, padding=5)  # 去掉 LabelFrame
        info_frame.pack(fill=tk.X, padx=5, pady=2)

        # 1. 先放置按钮容器到右侧，确保其不被裁切
        btn_frame = ttk.Frame(info_frame)
        btn_frame.pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="基本信息表", command=self.show_basic_table).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_frame, text="帧/流星表", command=self.show_meteor_table).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_frame, text="恒星表", command=self.show_star_table).pack(
            side=tk.LEFT, padx=2
        )

        # 2. 左侧放置两行文本信息
        text_info_frame = ttk.Frame(info_frame)
        text_info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.info_name_var = tk.StringVar(value="未加载数据")
        self.info_detail_var = tk.StringVar(value="")

        # 第一行：流星名称 (加粗)
        lbl_name = ttk.Label(
            text_info_frame,
            textvariable=self.info_name_var,
            font=(self.custom_font.actual("family"), 10, "bold"),
        )
        lbl_name.pack(side=tk.TOP, anchor="w")

        # 第二行：站点、相机、时间
        lbl_detail = ttk.Label(
            text_info_frame, textvariable=self.info_detail_var, font=self.custom_font
        )
        lbl_detail.pack(side=tk.TOP, anchor="w")

        # === 区域3: 播放器控制与显示 ===
        player_frame = ttk.Frame(self.root)
        player_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 3.1 叠加层控制
        overlay_ctrl = ttk.Frame(player_frame)
        overlay_ctrl.pack(fill=tk.X)

        self.show_signal = tk.BooleanVar(value=True)
        self.show_mask = tk.BooleanVar(value=True)
        self.show_stars = tk.BooleanVar(value=True)
        self.show_time = tk.BooleanVar(value=True)
        self.show_hist = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            overlay_ctrl,
            text="流星区域",
            variable=self.show_signal,
            command=self.refresh_canvas,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            overlay_ctrl,
            text="遮罩区域",
            variable=self.show_mask,
            command=self.refresh_canvas,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            overlay_ctrl,
            text="恒星标记",
            variable=self.show_stars,
            command=self.refresh_canvas,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            overlay_ctrl,
            text="时间戳",
            variable=self.show_time,
            command=self.refresh_canvas,
        ).pack(side=tk.LEFT)

        # 3.2 进度与播放控制
        play_ctrl = ttk.Frame(player_frame)
        play_ctrl.pack(fill=tk.X, pady=5)

        self.time_label = ttk.Label(play_ctrl, text="00:00:00", width=10)
        self.time_label.pack(side=tk.LEFT)

        self.frame_label = ttk.Label(play_ctrl, text="0/0", width=10)
        self.frame_label.pack(side=tk.LEFT)

        ttk.Button(play_ctrl, text="▶", width=3, command=self.toggle_play).pack(
            side=tk.LEFT
        )
        ttk.Button(play_ctrl, text="■", width=3, command=self.stop_play).pack(
            side=tk.LEFT
        )

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Scale(
            play_ctrl,
            orient="horizontal",
            variable=self.progress_var,
            command=self.on_seek,
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        ttk.Label(play_ctrl, text="速度:").pack(side=tk.LEFT)
        self.speed_combo = ttk.Combobox(
            play_ctrl, values=["1.0x", "0.5x", "0.1x"], width=5, state="readonly"
        )
        self.speed_combo.current(0)
        self.speed_combo.pack(side=tk.LEFT)
        self.speed_combo.bind("<<ComboboxSelected>>", self.change_speed)

        ttk.Checkbutton(
            play_ctrl,
            text="直方图",
            variable=self.show_hist,
            command=self.toggle_histogram,
        ).pack(side=tk.LEFT, padx=5)

        # 3.3 图像 Canvas
        self.canvas_frame = ttk.Frame(player_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="black", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 绑定事件
        self.canvas.bind("<MouseWheel>", self.on_scroll)
        self.canvas.bind("<Control-MouseWheel>", self.on_zoom)  # Windows
        self.canvas.bind("<ButtonPress-1>", self.on_pan_start)  # 左键按下
        self.canvas.bind("<B1-Motion>", self.on_pan_move)  # 左键拖拽
        self.canvas.bind("<Double-Button-1>", self.on_double_click)  # 新增：左键双击
        self.canvas.bind("<Configure>", self.on_resize)  # 窗口大小改变

    def open_fits(self):
        path = filedialog.askopenfilename(
            filetypes=[("FITS files", "*.fits"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            self.obs = MeteorObservation.from_fits(path)
            self.load_data()
        except Exception as e:
            messagebox.showerror("错误", f"打开文件失败:\n{str(e)}")
            import traceback

            traceback.print_exc()

    def save_fits(self):
        if not self.obs:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".fits", filetypes=[("FITS files", "*.fits")]
        )
        if path:
            try:
                self.obs.to_fits(path)
                messagebox.showinfo("成功", "文件保存成功")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败:\n{str(e)}")

    def load_data(self):
        """数据加载后的初始化"""
        # 1. 更新 contents 标签
        for widget in self.contents_frame.winfo_children():
            widget.destroy()

        possible_contents = [
            "image",
            "time",
            "star",
            "meteor",
            "calibration",
            "photometry",
        ]
        for c in possible_contents:
            bg_color = "#90EE90" if c in self.obs.contents else "#FFB6C1"  # 浅绿 / 浅红
            lbl = tk.Label(
                self.contents_frame,
                text=c,
                bg=bg_color,
                padx=5,
                pady=2,
                font=("Arial", 9, "bold"),
            )
            lbl.pack(side=tk.LEFT, padx=2)

        # 2. 更新 Info
        self.info_name_var.set(f"流星名称: {self.obs.meteor_name or 'N/A'}")
        detail_text = (
            f"站点: {self.obs.station_name} | 相机: {self.obs.camera_name} | "
            f"时间: {Time(self.obs.mean_time, format='unix').isot if self.obs.mean_time else 'N/A'}"
        )
        self.info_detail_var.set(detail_text)

        # 3. 重置播放器状态
        self.current_frame_idx = 0
        self.progress_bar.config(
            to=self.obs.frame_count - 1 if self.obs.frame_count else 0
        )

        # 4. 预计算/缓存一些辅助数据
        self.signal_edges = ImageProcessor.get_edges(self.obs.signal_frame)

        # 5. 检测数据类型并设置直方图范围 (兼容 uint16, float32)
        if self.obs.max_frame is not None:
            max_val = np.max(self.obs.max_frame)
            self.hist_window.update_range_limits(self.obs.max_frame.dtype.type, max_val)
            self.hist_window.set_data(self.obs.max_frame)
        elif self.obs.data is not None:
            max_val = np.max(self.obs.data[0])  # 简单取第一帧估计
            self.hist_window.update_range_limits(self.obs.data.dtype.type, max_val)

        # 6. 初始化视图：自适应窗口
        self.reset_view()

        self.update_image_display()

    def reset_view(self, event=None):
        """将画面缩放至适合当前窗口，并居中"""
        if not self.obs:
            return

        # 强制更新窗口以获取正确的 Canvas 尺寸
        self.root.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        # 获取图像原始尺寸（以 max_frame 或数据帧为准）
        img_w = self.obs.image_width
        img_h = self.obs.image_height

        if img_w > 0 and img_h > 0:
            # 计算缩放比例，取宽和高中较小的那个，确保不超出
            scale_w = cw / img_w
            scale_h = ch / img_h
            self.zoom_scale = min(scale_w, scale_h) * 0.95  # 留 5% 的边距

            # 重置偏移量（居中）
            self.pan_offset_x = 0
            self.pan_offset_y = 0

            self.update_image_display()

    # --- 播放逻辑 ---

    def toggle_play(self):
        if self.obs is None or self.obs.data is None:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_loop()

    def stop_play(self):
        self.is_playing = False
        self.current_frame_idx = 0
        self.update_image_display()

    def play_loop(self):
        if not self.is_playing:
            return

        # 播放到末尾循环
        if self.current_frame_idx >= self.obs.frame_count - 1:
            self.current_frame_idx = 0
        else:
            self.current_frame_idx += 1

        self.update_image_display()
        self.root.after(int(self.fps_delay), self.play_loop)

    def change_speed(self, event):
        val = self.speed_combo.get()
        speed = float(val.replace("x", ""))
        self.fps_delay = 40 / speed  # 基础40ms

    def on_seek(self, val):
        self.current_frame_idx = int(float(val))
        self.update_image_display()

    # --- 图像显示核心 ---

    def refresh_canvas(self):
        """只重绘 Canvas 上的内容，不重新计算图像数据（如果没变）"""
        self.update_image_display(recompute_img=False)

    def update_image_display(
        self, vmin=None, vmax=None, gamma=None, recompute_img=True
    ):
        if not self.obs:
            return

        # 获取直方图参数
        if vmin is None:
            vmin = self.hist_window.vmin
        if vmax is None:
            vmax = self.hist_window.vmax
        if gamma is None:
            gamma = self.hist_window.gamma

        # 1. 确定要显示的数据源
        if self.is_playing or (
            self.current_frame_idx > 0
            and self.current_frame_idx < (self.obs.frame_count or 0)
        ):
            if self.obs.data is not None:
                raw_data = self.obs.data[self.current_frame_idx]
            else:
                raw_data = np.zeros((self.obs.image_height, self.obs.image_width))
        else:
            # 停止状态显示 max_frame
            raw_data = (
                self.obs.max_frame
                if self.obs.max_frame is not None
                else np.zeros((100, 100))
            )

        # 2. 数值处理 -> 8bit Image (仅当需要时)
        if recompute_img:
            # 步骤 2a: 处理 Bayer 或 BGR 模式
            # 放在 apply_levels 之前以尽量利用原始数据的动态范围进行插值
            color_corrected = ImageProcessor.debayer_and_color_correct(
                raw_data, self.obs.color
            )
            if self.show_hist.get():
                self.hist_window.set_data(color_corrected)
            # 步骤 2b: 色阶映射到 8bit
            img_8bit = ImageProcessor.apply_levels(color_corrected, vmin, vmax, gamma)

            # 步骤 2c: 转为 PIL
            if img_8bit.ndim == 3:
                self.pil_image = Image.fromarray(img_8bit, mode="RGB")
            else:
                self.pil_image = Image.fromarray(img_8bit, mode="L")

        if not hasattr(self, "pil_image"):
            return

        # 3. 应用缩放与平移 (View Transform)
        # 强制更新以防获取到 1x1 的初始尺寸
        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        # 目标显示尺寸
        orig_w, orig_h = self.pil_image.size
        new_w = int(orig_w * self.zoom_scale)
        new_h = int(orig_h * self.zoom_scale)

        # 创建显示画布
        display_img = Image.new("RGB", (cw, ch), (0, 0, 0))

        # Resize 原图
        # 如果图极大，resize 会卡。为了演示，我们假设图不大 (通常流星监控 < 4K)
        # 优化: nearest neighbor 速度快
        resized_pil = self.pil_image.resize((new_w, new_h), Image.NEAREST)

        # 确保 resized_pil 是 RGB 方便后续合成
        if resized_pil.mode != "RGB":
            resized_pil = resized_pil.convert("RGB")

        if self.show_mask.get() and self.obs.mask_frame is not None:
            # 逻辑：mask_frame 为 0 表示无效区域（需变暗），1 为有效
            # 生成 Alpha 通道：0 的地方是不透明(160)，1 的地方是透明(0)
            mask_data = (self.obs.mask_frame == 0).astype(np.uint8) * 160

            # 将 Mask 缩放到当前显示尺寸
            mask_alpha = Image.fromarray(mask_data, mode="L").resize(
                (new_w, new_h), Image.NEAREST
            )

            # 创建黑色层 (尺寸与 resized_pil 一致)
            black_overlay = Image.new("RGB", (new_w, new_h), (0, 0, 0))

            # 将黑色层贴在 resized_pil 上 (坐标 0,0 绝对对齐)
            resized_pil.paste(black_overlay, (0, 0), mask=mask_alpha)

        # 粘贴位置
        paste_x = int(cw / 2 - new_w / 2 + self.pan_offset_x)
        paste_y = int(ch / 2 - new_h / 2 + self.pan_offset_y)

        # 粘贴 (支持负坐标，需计算裁切)
        box = (paste_x, paste_y)
        display_img.paste(resized_pil, box)

        # 4. 处理叠加层 (在 PIL 上画或者 Canvas 上画)
        # draw = ImageDraw.Draw(display_img)

        # 坐标变换函数: Image coords (ix, iy) -> Canvas coords (cx, cy)
        def to_canvas(ix, iy):
            cx = ix * self.zoom_scale + paste_x
            cy = iy * self.zoom_scale + paste_y
            return cx, cy

        # 4.2 流星区域 (蓝色轮廓)
        if self.show_signal.get() and self.signal_edges is not None:
            # 将 edges 缩放
            edge_pil = Image.fromarray(
                (self.signal_edges).astype(np.uint8) * 255
            ).resize((new_w, new_h), Image.NEAREST)
            # 将边缘染成蓝色粘贴
            blue_layer = Image.new("RGB", (new_w, new_h), (0, 100, 255))
            display_img.paste(blue_layer, (paste_x, paste_y), mask=edge_pil)

        # 转为 ImageTk
        self.img_tk = ImageTk.PhotoImage(display_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

        # 4.3 恒星标记 (Canvas 绘制)
        if self.show_stars.get() and self.obs.star_pixel_coord is not None:
            sx, sy = self.obs.star_pixel_coord
            names = self.obs.star_name
            r_base = 5
            r = (
                r_base * self.zoom_scale
            )  # 圈随放大而放大? 需求: "圈直径对应实际画面的约5像素" -> 即随缩放变大

            # 优化: 只绘制在屏幕范围内的
            for i in range(len(sx)):
                cx, cy = to_canvas(sx[i], sy[i])
                if -20 < cx < cw + 20 and -20 < cy < ch + 20:
                    self.canvas.create_oval(
                        cx - r, cy - r, cx + r, cy + r, outline="#00FF00", width=1
                    )
                    if names and i < len(names):
                        # 获取 HIP 编号并转换为字符串作为字典键
                        hip_id = str(names[i])
                        # 在对照表中查找，找不到则保留原 HIP 编号
                        display_name = self.star_common_names.get(hip_id, hip_id)

                        self.canvas.create_text(
                            cx + r + 2,
                            cy,
                            text=display_name,  # 使用真实名字或原编号
                            fill="#00FF00",
                            anchor="w",
                            font=self.custom_font,
                        )

        # 4.4 时间戳
        if self.show_time.get():
            t_str = "TIME: N/A"
            # 判断逻辑与上方 raw_data 选取逻辑保持一致
            is_showing_frame = self.is_playing or (
                self.current_frame_idx > 0
                and self.current_frame_idx < (self.obs.frame_count or 0)
            )

            if is_showing_frame:
                # 计算并显示当前显示帧的时间
                t_val = (
                    self.obs.frame_time[self.current_frame_idx]
                    if self.obs.frame_time is not None
                    else 0
                )
                if self.obs.mean_time:
                    abs_time = self.obs.mean_time + t_val
                    t_str = Time(abs_time, format="unix").isot
            else:
                # 处于停止状态显示 max_frame 时，显示平均时间
                t_str = f"MeanT: {Time(self.obs.mean_time, format='unix').isot if self.obs.mean_time else 'N/A'}"

            self.canvas.create_text(
                10,
                10,
                text=t_str,
                fill="white",
                anchor="nw",
                font=("Courier", 12, "bold"),
            )

        # 更新进度条 UI
        self.progress_var.set(self.current_frame_idx)
        self.frame_label.config(
            text=f"{self.current_frame_idx}/{self.obs.frame_count-1}"
        )

    # --- 交互事件 ---

    def on_scroll(self, event):
        """鼠标滚轮切换帧"""
        if self.is_playing:
            self.is_playing = False  # 仅停止播放，不重置索引到0

        delta = -1 if event.delta > 0 else 1  # Windows delta usually 120
        new_idx = self.current_frame_idx + delta
        if 0 <= new_idx < (self.obs.frame_count or 0):
            self.current_frame_idx = new_idx
            self.update_image_display()

    def on_zoom(self, event):
        """Ctrl+滚轮缩放"""
        if not self.obs:
            return
        scale_factor = 1.1 if event.delta > 0 else 0.9

        # 1. 记录鼠标在 Canvas 的绝对位置
        mx, my = event.x, event.y

        # 获取当前 Canvas 尺寸，用于对齐 update_image_display 中的中心算法
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        # 2. 计算缩放倍率
        old_scale = self.zoom_scale
        new_scale = old_scale * scale_factor
        if new_scale < 0.1:
            new_scale = 0.1
        if new_scale > 40:
            new_scale = 40  # 稍微放宽上限

        # 3. 核心算法：保持鼠标下的像素点不动
        # 鼠标相对于图像渲染中心的距离 (即 update_image_display 中的偏移基准)
        # 在 update_image_display 中，图像位置是 cw/2 + pan_offset
        # 所以鼠标相对于图像“原点”的距离是： mx - (cw/2 + self.pan_offset_x)

        ratio = new_scale / old_scale

        # 更新偏移量：新偏移 = 鼠标位置 - (鼠标到中心的旧距离 * 缩放比例) - Canvas中心
        # 简化后的公式如下：
        self.pan_offset_x = mx - cw / 2 - (mx - cw / 2 - self.pan_offset_x) * ratio
        self.pan_offset_y = my - ch / 2 - (my - ch / 2 - self.pan_offset_y) * ratio

        self.zoom_scale = new_scale
        self.update_image_display()

    def on_double_click(self, event):
        """双击左键恢复自适应"""
        self.reset_view()

    def on_pan_start(self, event):
        self.drag_start = (event.x, event.y)

    def on_pan_move(self, event):
        if not self.drag_start:
            return
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        self.pan_offset_x += dx
        self.pan_offset_y += dy
        self.drag_start = (event.x, event.y)
        self.update_image_display()

    def on_resize(self, event):
        # 简单防抖或直接重绘
        self.update_image_display(recompute_img=False)

    def toggle_histogram(self):
        if self.show_hist.get():
            self.hist_window.deiconify()
            # 刷新一下数据
            if self.obs:
                data = (
                    self.obs.data[self.current_frame_idx]
                    if self.is_playing and self.obs.data is not None
                    else self.obs.max_frame
                )
                if data is not None:
                    # 获取当前要显示的图像数据（可能是 Debayer 后的）
                    # 为了直方图准确，我们应该显示处理完颜色后的数据，但还没做 apply_levels 的数据
                    color_data = ImageProcessor.debayer_and_color_correct(
                        data, self.obs.color
                    )
                    self.hist_window.set_data(color_data)
        else:
            self.hist_window.withdraw()

    # --- 表格窗口调用 ---

    def show_basic_table(self):
        if not self.obs:
            return
        data = [
            ("版本", self.obs.version),
            ("数据模式", self.obs.mode),
            ("站点", self.obs.station_name),
            ("相机", self.obs.camera_name),
            ("位置", str(self.obs.station_location)),
            ("颜色模式", self.obs.color),
            ("处理时间", self.obs.processed_time),
            ("图像尺寸", f"{self.obs.image_width} x {self.obs.image_height}"),
            ("帧数", self.obs.frame_count),
            ("数据内容", ", ".join(self.obs.contents)),
            ("帧数", self.obs.frame_count),
            (
                "平均时间",
                (
                    Time(self.obs.mean_time, format="unix").isot
                    if self.obs.mean_time
                    else "N/A"
                ),
            ),
            ("流星名称", self.obs.meteor_name),
            ("时间方法", self.obs.time_method),
            ("时间来源", self.obs.time_source),
            ("寻星方法", self.obs.star_method),
            ("恒星数量", len(self.obs.star_name) if self.obs.star_name else 0),
            ("目标定位方法", self.obs.object_method),
            ("目标数量", len(self.obs.objects) if self.obs.objects else 0),
            ("相机校准方法", self.obs.calibration_method),
            ("相机校准结果", self.obs.calibration_result),
            ("相机校准残差", self.obs.calibration_residual),
            ("测光方法", self.obs.photometry_method),
            ("测光残差", self.obs.photometry_residual),
            ("数据库站点ID", self.obs.station_id),
            ("数据库流星ID", self.obs.meteor_id),
        ]
        InfoWindow(self.root, "基本信息", ["属性", "值"], data)

    def show_meteor_table(self):
        if not self.obs:
            return
        # 构建复杂的帧列表
        # 列: FrameIdx, Time, Obj1_X, Obj1_Y, Obj1_Mag, ...
        cols = ["Frame", "Time Offset"]
        if self.obs.objects:
            for i, obj in enumerate(self.obs.objects):
                cols.extend(
                    [
                        f"目标{i} Pixel X",
                        f"目标{i} Pixel Y",
                        f"目标{i} ECI X",
                        f"目标{i} ECI Y",
                        f"目标{i} ECI Z",
                        f"目标{i} 流量",
                        f"目标{i} 星等",
                    ]
                )

        rows = []
        for f_idx in range(self.obs.frame_count):
            t_off = self.obs.frame_time[f_idx] if self.obs.frame_time is not None else 0
            row = [f_idx, f"{t_off:.3f}"]

            if not self.obs.objects:
                rows.append(row)
                continue
            # 查找该帧下的流星数据
            for obj in self.obs.objects:
                # 这是一个低效查找，实际应该预处理 map
                # obj.meteor_index 是一个 list，包含该对象存在的帧索引
                found = False
                if obj.meteor_index is not None:
                    try:
                        # 找到 f_idx 在 meteor_index 中的位置
                        idx_in_obj = list(obj.meteor_index).index(f_idx)
                        px = obj.meteor_pixel_coord[0][idx_in_obj]
                        py = obj.meteor_pixel_coord[1][idx_in_obj]
                        ex = obj.meteor_eci_coord[0][idx_in_obj]
                        ey = obj.meteor_eci_coord[1][idx_in_obj]
                        ez = obj.meteor_eci_coord[2][idx_in_obj]
                        flux = (
                            obj.meteor_flux[idx_in_obj]
                            if obj.meteor_flux is not None
                            else ""
                        )
                        mag = (
                            obj.meteor_magnitude[idx_in_obj]
                            if obj.meteor_magnitude is not None
                            else ""
                        )
                        row.extend(
                            [
                                f"{px:.1f}",
                                f"{py:.1f}",
                                f"{ex:.5f}",
                                f"{ey:.5f}",
                                f"{ez:.5f}",
                                f"{flux:.2f}" if flux else "",
                                f"{mag:.2f}" if mag else "",
                            ]
                        )
                        found = True
                    except ValueError:
                        pass

                if not found:
                    row.extend(["", "", "", "", "", "", ""])
            rows.append(row)

        InfoWindow(self.root, "帧与流星数据", cols, rows)

    def show_star_table(self):
        if not self.obs or self.obs.star_name is None:
            return
        cols = [
            "HIP编号",
            "名称",
            "Pixel X",
            "Pixel Y",
            "ECI X",
            "ECI Y",
            "ECI Z",
            "星等",
            "流量",
        ]
        rows = []
        for i in range(len(self.obs.star_name)):
            hip_id = self.obs.star_name[i]
            real_name = self.star_common_names.get(str(hip_id), "")
            px = self.obs.star_pixel_coord[0][i]
            py = self.obs.star_pixel_coord[1][i]
            ex = (
                self.obs.star_eci_coord[0][i]
                if self.obs.star_eci_coord is not None
                else ""
            )
            ey = (
                self.obs.star_eci_coord[1][i]
                if self.obs.star_eci_coord is not None
                else ""
            )
            ez = (
                self.obs.star_eci_coord[2][i]
                if self.obs.star_eci_coord is not None
                else ""
            )
            mag = (
                self.obs.star_magnitude[i]
                if self.obs.star_magnitude is not None
                else ""
            )
            flux = self.obs.star_flux[i] if self.obs.star_flux is not None else ""
            rows.append(
                [
                    hip_id,
                    real_name,
                    f"{px:.1f}",
                    f"{py:.1f}",
                    f"{ex:.5f}",
                    f"{ey:.5f}",
                    f"{ez:.5f}",
                    flux,
                    mag,
                ]
            )

        InfoWindow(self.root, "恒星表", cols, rows)


def main():
    root = tk.Tk()
    # 设置一些全局样式
    style = ttk.Style()
    style.theme_use("clam")  # 比较现代的风格
    app = MeteorPlayerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
