# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
import os
import json
import threading
import xml.etree.ElementTree as ET
from PIL import Image, ImageTk
from datetime import datetime, timedelta
from astropy.time import Time
from .observation import MeteorObservation
from importlib import resources


def get_resource_path(package_data, resource_name):
    return str(resources.files(package_data).joinpath(resource_name))


# 尝试引入天文学库
HAS_SKYFIELD = False
try:
    from skyfield.api import Star, load
    from skyfield.data import hipparcos

    HAS_SKYFIELD = True
except ImportError:
    print("Warning: Skyfield not found. ECI calculation will be disabled.")

# --- 常量定义 ---
COLOR_MASK_DISPLAY = (255, 0, 0, 100)  # 显示用：红色，带Alpha
COLOR_SIGNAL_DISPLAY = (0, 100, 255, 100)  # 显示用：蓝色，带Alpha
DIFF_THRESHOLD = 2  # 帧去重：像素差值阈值 (X)
DIFF_RATIO = 0.01  # 帧去重：差异像素占比阈值 (Y)


class EditorCanvas(tk.Canvas):
    """
    继承自 Canvas，处理缩放、平移、绘制逻辑
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.config(bg="black", cursor="cross")

        # 视图状态
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0

        # 数据引用
        self.bg_image = None  # PIL Image (Base)
        self.mask_data = None  # Numpy uint8 (0 or 1)
        self.signal_data = None  # Numpy uint8 (0 or 1)
        self.star_list = []  # [{'id': 'a', 'x': 100, 'y': 100, 'name': ''}, ...]

        # 交互模式: 'view', 'mask', 'signal', 'star'
        self.mode = "view"
        self.brush_size = 10
        self.is_eraser = False

        # 缓存图像以防GC
        self.tk_img = None

        # 鼠标事件绑定
        self.bind("<ButtonPress-1>", self.on_left_down)
        self.bind("<B1-Motion>", self.on_left_drag)
        self.bind("<ButtonRelease-1>", self.on_left_up)

        # 中键/右键平移
        self.bind("<ButtonPress-2>", self.on_pan_down)
        self.bind("<B2-Motion>", self.on_pan_drag)
        self.bind("<ButtonRelease-2>", self.on_pan_up)
        self.bind("<ButtonPress-3>", self.on_pan_down)
        self.bind("<B3-Motion>", self.on_pan_drag)
        self.bind("<ButtonRelease-3>", self.on_pan_up)

        # 滚轮事件
        self.bind("<MouseWheel>", self.on_wheel)  # Windows
        self.bind("<Button-4>", self.on_wheel)  # Linux scroll up
        self.bind("<Button-5>", self.on_wheel)  # Linux scroll down
        self.bind("<Control-MouseWheel>", self.on_zoom)  # Zoom
        self.bind("<Double-Button-1>", self.on_double_click)

        self.last_mouse_draw = None
        self.last_mouse_pan = None

        # 回调函数接口
        self.on_star_add = None
        self.on_frame_scroll = None
        self.on_edit_action = None  # 当发生绘制操作时触发，用于更新状态

    def set_image(self, numpy_img):
        if numpy_img is None:
            self.bg_image = None
            return

        # 转换色彩空间用于显示
        if len(numpy_img.shape) == 3:
            rgb = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(numpy_img, cv2.COLOR_GRAY2RGB)

        self.bg_image = Image.fromarray(rgb)
        self.redraw()

    def fit_to_window(self):
        if self.bg_image is None:
            return
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        iw, ih = self.bg_image.size

        if w < 10 or h < 10:
            return

        scale_w = w / iw
        scale_h = h / ih
        self.zoom_scale = min(scale_w, scale_h) * 0.95
        self.pan_x = 0
        self.pan_y = 0
        self.redraw()

    def canvas_to_image(self, cx, cy):
        """将画布坐标转换为图像像素坐标"""
        if not self.bg_image:
            return 0, 0

        w = int(self.winfo_width())
        h = int(self.winfo_height())
        img_w, img_h = self.bg_image.size

        disp_w = int(img_w * self.zoom_scale)
        disp_h = int(img_h * self.zoom_scale)

        offset_x = (w - disp_w) / 2 + self.pan_x
        offset_y = (h - disp_h) / 2 + self.pan_y

        ix = (cx - offset_x) / self.zoom_scale
        iy = (cy - offset_y) / self.zoom_scale
        return ix, iy

    def redraw(self):
        self.delete("all")
        if self.bg_image is None:
            return

        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1 or h <= 1:
            return

        img_w, img_h = self.bg_image.size
        new_w = int(img_w * self.zoom_scale)
        new_h = int(img_h * self.zoom_scale)

        if new_w <= 0 or new_h <= 0:
            return

        # 1. 基础图像
        display_img = self.bg_image.resize((new_w, new_h), Image.NEAREST)

        # 2. 叠加 Mask (红色)
        if self.mask_data is not None:
            # 创建全红层
            mask_layer = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            mask_layer[:] = (255, 0, 0)  # RGB Red
            # Alpha 通道：mask为0的地方设为100，否则0
            alpha = ((1 - self.mask_data) * 100).astype(np.uint8)

            mask_pil = Image.fromarray(mask_layer)
            mask_pil.putalpha(Image.fromarray(alpha))
            mask_pil = mask_pil.resize((new_w, new_h), Image.NEAREST)
            display_img.paste(mask_pil, (0, 0), mask_pil)

        # 3. 叠加 Signal (蓝色)
        if self.signal_data is not None:
            sig_layer = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            sig_layer[:] = (0, 100, 255)  # RGB Blue-ish
            alpha = (self.signal_data * 100).astype(np.uint8)

            sig_pil = Image.fromarray(sig_layer)
            sig_pil.putalpha(Image.fromarray(alpha))
            sig_pil = sig_pil.resize((new_w, new_h), Image.NEAREST)
            display_img.paste(sig_pil, (0, 0), sig_pil)

        # 4. 创建最终画布图像
        final_img = Image.new("RGB", (w, h), (0, 0, 0))
        paste_x = int((w - new_w) / 2 + self.pan_x)
        paste_y = int((h - new_h) / 2 + self.pan_y)
        final_img.paste(display_img, (paste_x, paste_y))

        self.tk_img = ImageTk.PhotoImage(final_img)
        self.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        # 5. 绘制恒星标记
        for star in self.star_list:
            ix, iy = star["x"], star["y"]
            cx = ix * self.zoom_scale + paste_x
            cy = iy * self.zoom_scale + paste_y

            # 仅在视野内绘制
            if -20 < cx < w + 20 and -20 < cy < h + 20:
                r = 5
                # 有名字或ID视为已标记
                is_identified = bool(star.get("name"))
                color = "#00FF00" if is_identified else "#FFFF00"
                self.create_line(cx - r, cy, cx + r, cy, fill=color, width=1)
                self.create_line(cx, cy - r, cx, cy + r, fill=color, width=1)

                label_txt = star["id"]
                if is_identified:
                    # 简略显示名字，太长影响视野，这里只显示输入的名字（可能是数字）
                    label_txt += f" {star.get('name', '')}"

                self.create_text(
                    cx + r + 5,
                    cy,
                    text=label_txt,
                    fill=color,
                    anchor="w",
                    font=("Arial", 10, "bold"),
                )

    def on_left_down(self, event):
        self.last_mouse_draw = (event.x, event.y)
        if self.mode in ["mask", "signal"]:
            self.paint(event.x, event.y)
            if self.on_edit_action:
                self.on_edit_action()

    def on_left_drag(self, event):
        if self.last_mouse_draw is None:
            return
        if self.mode in ["mask", "signal"]:
            ix1, iy1 = self.canvas_to_image(
                self.last_mouse_draw[0], self.last_mouse_draw[1]
            )
            ix2, iy2 = self.canvas_to_image(event.x, event.y)
            self.paint_line(ix1, iy1, ix2, iy2)
            self.redraw()
            if self.on_edit_action:
                self.on_edit_action()
        self.last_mouse_draw = (event.x, event.y)

    def on_left_up(self, event):
        self.last_mouse_draw = None

    def on_pan_down(self, event):
        self.last_mouse_pan = (event.x, event.y)

    def on_pan_drag(self, event):
        if self.last_mouse_pan is None:
            return
        dx = event.x - self.last_mouse_pan[0]
        dy = event.y - self.last_mouse_pan[1]
        self.pan_x += dx
        self.pan_y += dy
        self.redraw()
        self.last_mouse_pan = (event.x, event.y)

    def on_pan_up(self, event):
        self.last_mouse_pan = None

    def on_double_click(self, event):
        if self.mode == "star" and self.on_star_add:
            ix, iy = self.canvas_to_image(event.x, event.y)
            if self.bg_image:
                w, h = self.bg_image.size
                if 0 <= ix < w and 0 <= iy < h:
                    self.on_star_add(ix, iy)
        elif self.mode == "view":
            self.fit_to_window()

    def on_wheel(self, event):
        if self.on_frame_scroll:
            direction = 0
            if event.num == 4 or event.delta > 0:
                direction = -1
            if event.num == 5 or event.delta < 0:
                direction = 1
            self.on_frame_scroll(direction)

    def on_zoom(self, event):
        scale_factor = 1.1 if event.delta > 0 else 0.9
        self.zoom_scale *= scale_factor
        self.zoom_scale = max(0.05, min(50, self.zoom_scale))
        self.redraw()

    def paint(self, cx, cy):
        ix, iy = self.canvas_to_image(cx, cy)
        self.paint_line(ix, iy, ix, iy)
        self.redraw()

    def paint_line(self, x1, y1, x2, y2):
        target = self.mask_data if self.mode == "mask" else self.signal_data
        if target is None:
            return
        # 根据模式决定绘制的值
        if self.mode == "mask":
            # Mask模式：画笔遮挡设为0，橡皮擦还原设为1
            val = 1 if self.is_eraser else 0
        else:
            # Signal模式：画笔标记设为1，橡皮擦擦除设为0
            val = 0 if self.is_eraser else 1
        cv2.line(
            target,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            val,
            thickness=self.brush_size,
        )
        cv2.circle(target, (int(x2), int(y2)), self.brush_size // 2, val, -1)


class StarDialog(tk.Toplevel):
    def __init__(self, parent, star_list, update_cb, names_dict, id_map, validate_cb):
        super().__init__(parent)
        self.title("恒星信息表")
        self.geometry("920x450")
        self.star_list = star_list
        self.update_cb = update_cb
        self.names_dict = names_dict
        self.id_map = id_map  # 反向映射表 {HIP_ID: Name}
        self.validate_cb = validate_cb  # 验证回调函数
        self.all_names = sorted(list(self.names_dict.keys()))
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # 列表区域
        cols = ("ID", "恒星 (HIP - 名)", "X", "Y", "ECI_X", "ECI_Y", "ECI_Z")
        self.tree = ttk.Treeview(self, columns=cols, show="headings")
        for col in cols:
            self.tree.heading(col, text=col)
            if "ECI" in col:
                self.tree.column(col, width=90)
            elif col == "ID":
                self.tree.column(col, width=40)
            elif "恒星" in col:
                self.tree.column(col, width=150)
            else:
                self.tree.column(col, width=60)

        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        # 编辑区域
        edit_frame = ttk.Frame(self)
        edit_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(edit_frame, text="恒星名/HIP编号:").pack(side=tk.LEFT)
        self.var_name = tk.StringVar()
        self.cb_name = ttk.Combobox(
            edit_frame, textvariable=self.var_name, values=self.all_names, width=20
        )
        self.cb_name.pack(side=tk.LEFT, padx=5)
        self.cb_name.bind("<KeyRelease>", self.filter_names)
        self.cb_name.bind("<<ComboboxSelected>>", self.save_edit)
        self.cb_name.bind("<Return>", self.save_edit)

        ttk.Label(edit_frame, text="X:").pack(side=tk.LEFT)
        self.var_x = tk.DoubleVar()
        self.entry_x = ttk.Entry(edit_frame, textvariable=self.var_x, width=8)
        self.entry_x.pack(side=tk.LEFT)

        ttk.Label(edit_frame, text="Y:").pack(side=tk.LEFT)
        self.var_y = tk.DoubleVar()
        self.entry_y = ttk.Entry(edit_frame, textvariable=self.var_y, width=8)
        self.entry_y.pack(side=tk.LEFT)

        ttk.Button(edit_frame, text="保存&计算", command=self.save_edit).pack(
            side=tk.LEFT, padx=10
        )
        ttk.Button(edit_frame, text="删除选中行", command=self.delete_row).pack(
            side=tk.LEFT, padx=10
        )
        self.lbl_status = ttk.Label(edit_frame, text="", foreground="red")
        self.lbl_status.pack(side=tk.LEFT, padx=5)

        self.refresh_table()

    def filter_names(self, event):
        val = event.widget.get().strip()
        # 如果输入的是数字，不进行中文名过滤，保持原样以便输入HIP ID
        if val.isdigit():
            return
        if val:
            self.cb_name["values"] = [x for x in self.all_names if val in x]
        else:
            self.cb_name["values"] = self.all_names

    def refresh_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for s in self.star_list:
            ex = f"{s.get('eci_x', 0):.4f}"
            ey = f"{s.get('eci_y', 0):.4f}"
            ez = f"{s.get('eci_z', 0):.4f}"

            # --- 格式化显示逻辑 ---
            raw_input = s.get("name", "").strip()
            display_text = raw_input

            hip_id = None
            resolved_name = ""

            # 判断 raw_input 是名字还是数字
            if raw_input.isdigit():
                hip_id = int(raw_input)
                # 尝试反查名字
                resolved_name = self.id_map.get(hip_id, "")
            elif raw_input in self.names_dict:
                hip_id = self.names_dict[raw_input]
                resolved_name = raw_input  # 用户输入的就是名字

            # 构建显示字符串
            if hip_id is not None:
                if resolved_name and resolved_name != str(hip_id):
                    display_text = f"HIP {hip_id} - {resolved_name}"
                else:
                    display_text = f"HIP {hip_id}"

            self.tree.insert(
                "",
                "end",
                values=(
                    s["id"],
                    display_text,
                    f"{s['x']:.2f}",
                    f"{s['y']:.2f}",
                    ex,
                    ey,
                    ez,
                ),
            )

    def on_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        # 获取原始数据中的name，而不是表格中格式化后的文字
        idx = self.tree.index(sel[0])
        if idx < len(self.star_list):
            s = self.star_list[idx]
            self.var_name.set(s.get("name", ""))
            self.var_x.set(s["x"])
            self.var_y.set(s["y"])

    def save_edit(self, event=None):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])

        # 获取输入内容，去除空格
        val = self.var_name.get().strip()

        # --- 验证逻辑 ---
        if self.validate_cb:
            is_valid, msg = self.validate_cb(val)
            if not is_valid:
                self.lbl_status.config(text=f"错误: {msg}", foreground="red")
                return  # 验证失败，终止保存，不修改数据
        # ----------------

        s = self.star_list[idx]

        try:
            new_x = float(self.var_x.get())
            new_y = float(self.var_y.get())
        except ValueError:
            self.lbl_status.config(text="错误: 坐标必须是数字", foreground="red")
            return

        # 验证通过后修改数据
        s["name"] = val
        s["x"] = new_x
        s["y"] = new_y

        # 触发外部更新（包括坐标计算）
        self.update_cb()
        self.refresh_table()

        # 重新选中该行
        new_items = self.tree.get_children()
        if idx < len(new_items):
            self.tree.selection_set(new_items[idx])

    def delete_row(self):
        sel = self.tree.selection()
        if not sel:
            return
        if hasattr(self, "lbl_status"):
            self.lbl_status.config(text="")
        idx = self.tree.index(sel[0])
        del self.star_list[idx]

        # 重新编号
        for i, s in enumerate(self.star_list):
            s["id"] = chr(ord("a") + i)

        self.refresh_table()
        self.update_cb()

    def on_close(self):
        self.destroy()


class ConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Meteor Observation Converter")
        self.root.geometry("1400x950")

        self.obs = MeteorObservation()
        self.video_path = None
        self.eff_fps = 25.0
        self.total_frames_original = 0

        # 状态标记
        self.flag_mask_edited = False
        self.flag_signal_edited = False

        self.star_win = None
        self.star_names_map = {}  # Name -> HIP
        self.star_id_to_name = {}  # HIP -> Name (优先中文)

        self.load_star_names()
        self.load_skyfield_data()

        self.setup_ui()

    def load_star_names(self):
        """加载恒星中文/英文名到 HIP 编号的映射，同时构建反向映射"""
        self.star_names_map = {}
        self.star_id_to_name = {}
        try:
            with open(
                get_resource_path("omdx.data", "star_common_names.json"),
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

                # 优先加载英文构建反向表
                for hip, name in data.get("English", {}).items():
                    hip_int = int(hip)
                    self.star_names_map[name] = hip_int
                    if hip_int not in self.star_id_to_name:
                        self.star_id_to_name[hip_int] = name

                # 加载中文，覆盖反向表中的英文名（如果存在），实现优先显示中文
                for hip, name in data.get("Chinese", {}).items():
                    hip_int = int(hip)
                    self.star_names_map[name] = hip_int
                    self.star_id_to_name[hip_int] = name

        except Exception:
            pass

    def load_skyfield_data(self):
        if HAS_SKYFIELD:
            threading.Thread(target=self._load_sf_thread, daemon=True).start()

    def _load_sf_thread(self):
        try:
            with load.open(get_resource_path("omdx.data", "hip_main.dat")) as f:
                self.hipparcos_df = hipparcos.load_dataframe(f)
            self.ts = load.timescale()
            self.planets = load(get_resource_path("omdx.data", "de421.bsp"))
            self.earth = self.planets["earth"]
        except Exception as e:
            print(f"Skyfield data load error: {e}")
            self.hipparcos_df = None

    def setup_ui(self):
        # 布局主框架
        left_panel = ttk.Frame(self.root, width=420, padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)

        right_panel = ttk.Frame(self.root, padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- 1. 文件操作 ---
        grp_file = ttk.LabelFrame(left_panel, text="1. 文件操作")
        grp_file.pack(fill=tk.X, pady=5)

        ttk.Button(grp_file, text="读取 UFO 视频", command=self.load_video).pack(
            fill=tk.X, pady=2
        )

        self.btn_save_fits = ttk.Button(
            grp_file,
            text="保存 FITS (to_fits)",
            command=self.save_fits,
            state="disabled",
        )
        self.btn_save_fits.pack(fill=tk.X, pady=2)

        self.btn_save_files = ttk.Button(
            grp_file,
            text="保存 Omdx (to_files)",
            command=self.save_files,
            state="disabled",
        )
        self.btn_save_files.pack(fill=tk.X, pady=2)

        # 状态指示灯
        self.status_frame = ttk.Frame(grp_file)
        self.status_frame.pack(fill=tk.X, pady=5)
        self.lbl_s_time = tk.Label(
            self.status_frame, text="TIME", bg="#FFB6C1", width=6, relief="groove"
        )
        self.lbl_s_time.pack(side=tk.LEFT, padx=2)
        self.lbl_s_image = tk.Label(
            self.status_frame, text="IMAGE", bg="#FFB6C1", width=6, relief="groove"
        )
        self.lbl_s_image.pack(side=tk.LEFT, padx=2)
        self.lbl_s_star = tk.Label(
            self.status_frame, text="STAR", bg="#FFB6C1", width=6, relief="groove"
        )
        self.lbl_s_star.pack(side=tk.LEFT, padx=2)

        # --- 2. 基础信息 ---
        grp_basic = ttk.LabelFrame(left_panel, text="2. 基础信息 (来自XML)")
        grp_basic.pack(fill=tk.X, pady=5)

        ttk.Label(
            grp_basic,
            text="规范配置UFOCapture软件后此栏信息可以自动生成",
            foreground="gray",
            font=("Arial", 8),
        ).pack(fill=tk.X, pady=(0, 5))

        self.var_m_name = tk.StringVar()
        self.var_s_name = tk.StringVar()
        self.var_c_name = tk.StringVar()
        self.var_lat = tk.StringVar(value="0")
        self.var_lon = tk.StringVar(value="0")
        self.var_alt = tk.StringVar(value="0")

        self._create_labeled_entry(grp_basic, "流星名称:", self.var_m_name)
        self._create_labeled_entry(grp_basic, "站点名称:", self.var_s_name)
        self._create_labeled_entry(grp_basic, "相机名称:", self.var_c_name)

        # 经纬高拆分
        loc_frame = ttk.Frame(grp_basic)
        loc_frame.pack(fill=tk.X, pady=2)
        ttk.Label(loc_frame, text="纬/经/高:", width=10).pack(side=tk.LEFT)
        ttk.Entry(loc_frame, textvariable=self.var_lat, width=8).pack(
            side=tk.LEFT, padx=1
        )
        ttk.Entry(loc_frame, textvariable=self.var_lon, width=8).pack(
            side=tk.LEFT, padx=1
        )
        ttk.Entry(loc_frame, textvariable=self.var_alt, width=8).pack(
            side=tk.LEFT, padx=1
        )

        # --- 3. 时间校准 ---
        grp_time = ttk.LabelFrame(left_panel, text="3. 时间校准")
        grp_time.pack(fill=tk.X, pady=5)

        ttk.Label(
            grp_time,
            text="请填写实际的时间戳：选择秒数变动的那一帧并填写时间",
            foreground="gray",
            font=("Arial", 8),
            wraplength=380,
        ).pack(fill=tk.X, pady=(0, 5))

        # FPS 显示
        f_fps = ttk.Frame(grp_time)
        f_fps.pack(fill=tk.X)
        self.var_fps = tk.DoubleVar(value=0.0)
        self.var_eff_fps = tk.DoubleVar(value=0.0)
        ttk.Label(f_fps, text="原始FPS:").pack(side=tk.LEFT)
        ttk.Label(f_fps, textvariable=self.var_fps).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_fps, text="有效FPS:").pack(side=tk.LEFT, padx=5)
        ttk.Label(f_fps, textvariable=self.var_eff_fps).pack(side=tk.LEFT)

        # 校准帧设定
        f_cal_frame = ttk.Frame(grp_time)
        f_cal_frame.pack(fill=tk.X, pady=2)
        ttk.Label(f_cal_frame, text="校准帧 (秒变动帧):").pack(side=tk.LEFT)
        self.var_cal_frame = tk.IntVar(value=0)
        self.spin_frame = ttk.Spinbox(
            f_cal_frame,
            from_=0,
            to=9999,
            textvariable=self.var_cal_frame,
            width=10,
            command=self.on_spin_seek,
        )
        self.spin_frame.pack(side=tk.LEFT, padx=5)

        # 时间输入
        f_date = ttk.Frame(grp_time)
        f_date.pack(fill=tk.X, pady=2)
        self.var_y, self.var_m, self.var_d = (
            tk.StringVar(),
            tk.StringVar(),
            tk.StringVar(),
        )
        self.var_hh, self.var_mm, self.var_ss = (
            tk.StringVar(),
            tk.StringVar(),
            tk.StringVar(),
        )

        f1 = ttk.Frame(f_date)
        f1.pack(fill=tk.X)
        ttk.Entry(f1, textvariable=self.var_y, width=5).pack(side=tk.LEFT)
        ttk.Label(f1, text="-").pack(side=tk.LEFT)
        ttk.Entry(f1, textvariable=self.var_m, width=3).pack(side=tk.LEFT)
        ttk.Label(f1, text="-").pack(side=tk.LEFT)
        ttk.Entry(f1, textvariable=self.var_d, width=3).pack(side=tk.LEFT)

        f2 = ttk.Frame(f_date)
        f2.pack(fill=tk.X, pady=2)
        ttk.Entry(f2, textvariable=self.var_hh, width=3).pack(side=tk.LEFT)
        ttk.Label(f2, text=":").pack(side=tk.LEFT)
        ttk.Entry(f2, textvariable=self.var_mm, width=3).pack(side=tk.LEFT)
        ttk.Label(f2, text=":").pack(side=tk.LEFT)
        ttk.Entry(f2, textvariable=self.var_ss, width=6).pack(side=tk.LEFT)

        ttk.Button(grp_time, text="应用时间", command=self.apply_time).pack(
            fill=tk.X, pady=2
        )

        # --- 4. 遮罩绘制 ---
        grp_draw = ttk.LabelFrame(left_panel, text="4. 遮罩绘制")
        grp_draw.pack(fill=tk.X, pady=5)

        self.mode_var = tk.StringVar(value="view")

        ttk.Radiobutton(
            grp_draw,
            text="浏览模式 (中键拖动)",
            variable=self.mode_var,
            value="view",
            command=self.on_mode_change,
        ).pack(anchor="w")

        # Mask 区域 + 按钮
        f_mask_tools = ttk.Frame(grp_draw)
        f_mask_tools.pack(fill=tk.X)
        ttk.Radiobutton(
            f_mask_tools,
            text="绘制遮罩 (红)",
            variable=self.mode_var,
            value="mask",
            command=self.on_mode_change,
        ).pack(side=tk.LEFT)
        ttk.Button(
            f_mask_tools, text="保存", width=4, command=self.save_mask_file
        ).pack(side=tk.RIGHT, padx=1)
        ttk.Button(
            f_mask_tools, text="加载", width=4, command=self.load_mask_file
        ).pack(side=tk.RIGHT, padx=1)

        ttk.Radiobutton(
            grp_draw,
            text="绘制流星区域 (蓝)",
            variable=self.mode_var,
            value="signal",
            command=self.on_mode_change,
        ).pack(anchor="w")

        # --- 5. 恒星信息 ---
        grp_star = ttk.LabelFrame(left_panel, text="5. 恒星信息")
        grp_star.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(
            grp_star,
            text="点选恒星 (绿)",
            variable=self.mode_var,
            value="star",
            command=self.on_mode_change,
        ).pack(anchor="w")
        ttk.Button(grp_star, text="打开恒星表", command=self.open_star_table).pack(
            fill=tk.X, pady=5
        )

        # --- 右侧：画布与播放控制 ---
        self.canvas_editor = EditorCanvas(right_panel)
        self.canvas_editor.on_star_add = self.add_star_point
        self.canvas_editor.on_frame_scroll = self.on_frame_scroll
        self.canvas_editor.on_edit_action = self.on_canvas_edited

        # 顶部工具栏
        tool_frame = ttk.Frame(right_panel)
        tool_frame.pack(fill=tk.X)
        ttk.Label(tool_frame, text="画笔大小:").pack(side=tk.LEFT)
        self.brush_scale = ttk.Scale(
            tool_frame,
            from_=1,
            to=50,
            length=100,
            command=lambda v: setattr(self.canvas_editor, "brush_size", int(float(v))),
        )
        self.brush_scale.set(10)
        self.brush_scale.pack(side=tk.LEFT, padx=5)

        self.is_erase_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            tool_frame,
            text="橡皮擦",
            variable=self.is_erase_var,
            command=lambda: setattr(
                self.canvas_editor, "is_eraser", self.is_erase_var.get()
            ),
        ).pack(side=tk.LEFT, padx=10)

        ttk.Button(tool_frame, text="清屏(当前层)", command=self.clear_layer).pack(
            side=tk.LEFT
        )
        ttk.Button(
            tool_frame, text="适应窗口", command=self.canvas_editor.fit_to_window
        ).pack(side=tk.LEFT, padx=10)

        self.canvas_editor.pack(fill=tk.BOTH, expand=True, pady=5)

        # 底部播放条
        play_ctrl = ttk.Frame(right_panel)
        play_ctrl.pack(fill=tk.X)
        self.btn_play = ttk.Button(
            play_ctrl, text="▶", width=3, command=self.toggle_play
        )
        self.btn_play.pack(side=tk.LEFT)
        self.btn_stop = ttk.Button(play_ctrl, text="■", width=3, command=self.stop_play)
        self.btn_stop.pack(side=tk.LEFT)

        self.curr_frame_var = tk.DoubleVar()
        self.scale_progress = ttk.Scale(
            play_ctrl,
            variable=self.curr_frame_var,
            from_=0,
            to=100,
            command=self.on_video_seek,
        )
        self.scale_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.lbl_frame_info = ttk.Label(play_ctrl, text="0/0")
        self.lbl_frame_info.pack(side=tk.LEFT)

        self.is_playing = False
        self.curr_frame_idx = 0

    def _create_labeled_entry(self, parent, label_text, var):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=2)
        ttk.Label(f, text=label_text, width=10).pack(side=tk.LEFT)
        ttk.Entry(f, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def load_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video", "*.avi;*.mp4"), ("All", "*.*")]
        )
        if path:
            threading.Thread(
                target=self._process_video_thread, args=(path,), daemon=True
            ).start()

    def _process_video_thread(self, path):
        try:
            cap = cv2.VideoCapture(path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_raw = cap.get(cv2.CAP_PROP_FPS)

            kept_frames = []
            prev_frame = None

            # 读取并去重
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 去重逻辑
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                is_duplicate = False

                if prev_frame is not None:
                    diff = cv2.absdiff(gray, prev_frame)
                    diff_count = np.count_nonzero(diff > DIFF_THRESHOLD)
                    diff_ratio = diff_count / gray.size
                    if diff_ratio < DIFF_RATIO:
                        is_duplicate = True

                if not is_duplicate:
                    kept_frames.append(frame)
                    prev_frame = gray

            cap.release()

            if not kept_frames:
                raise ValueError("Video has no frames.")

            # 初始化 Observation 对象
            self.obs.data = np.array(kept_frames)
            self.obs.frame_count = len(kept_frames)
            self.obs.image_height = self.obs.data[0].shape[0]
            self.obs.image_width = self.obs.data[0].shape[1]
            self.obs.color = "BGR"
            self.obs.mode = "fast"
            self.obs.meteor_name = os.path.splitext(os.path.basename(path))[0]
            self.obs.objects = []

            # 初始化 Mask 和 Signal
            self.obs.mask_frame = np.ones(
                (self.obs.image_height, self.obs.image_width), dtype=np.uint8
            )
            self.obs.signal_frame = np.zeros(
                (self.obs.image_height, self.obs.image_width), dtype=np.uint8
            )

            # 尝试自动加载 Signal (*M.bmp)
            bmp_path_guess_1 = os.path.splitext(path)[0] + "M.bmp"
            if os.path.exists(bmp_path_guess_1):
                self._load_signal_from_bmp(bmp_path_guess_1)

            # 处理 XML 元数据
            xml_path = os.path.splitext(path)[0] + ".xml"
            self.total_frames_original = total_frames

            if os.path.exists(xml_path):
                self._parse_xml(xml_path)
            else:
                self.var_fps.set(fps_raw)

            # 计算有效 FPS
            duration = (
                total_frames / self.var_fps.get() if self.var_fps.get() > 0 else 1.0
            )
            self.eff_fps = len(kept_frames) / duration if duration > 0 else 25.0
            self.var_eff_fps.set(round(self.eff_fps, 3))

            self.root.after(0, self._post_load_update)

        except Exception as e:
            self.root.after(
                0, lambda error=e: messagebox.showerror("Error", str(error))
            )

    def _load_signal_from_bmp(self, bmp_path):
        try:
            m_bmp = cv2.imread(bmp_path)
            if m_bmp is None:
                return

            # 确保尺寸一致
            if m_bmp.shape[:2] != (self.obs.image_height, self.obs.image_width):
                m_bmp = cv2.resize(m_bmp, (self.obs.image_width, self.obs.image_height))

            b, g, r = cv2.split(m_bmp)
            ret, thresh = cv2.threshold(r, 20, 1, cv2.THRESH_BINARY)

            # 形态学膨胀
            kernel = np.ones((10, 10), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)

            self.obs.signal_frame = dilated.astype(np.uint8)
            self.flag_signal_edited = True
            print(f"Loaded signal from {bmp_path}")
        except Exception as e:
            print(f"Failed to load M.bmp: {e}")

    def _parse_xml(self, path):
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            attr = root.attrib
            self.var_fps.set(float(attr.get("fps", 25.0)))
            self.var_s_name.set(attr.get("sid", ""))
            self.var_c_name.set(attr.get("cam", ""))

            self.var_lat.set(attr.get("lat", "0"))
            self.var_lon.set(attr.get("lng", "0"))
            self.var_alt.set(attr.get("alt", "0"))

            try:
                self.obs.station_location = (
                    float(attr.get("lat", 0)),
                    float(attr.get("lng", 0)),
                    float(attr.get("alt", 0)),
                )
            except Exception:
                pass
        except Exception as e:
            print(f"XML parse error: {e}")

    def _post_load_update(self):
        self.var_m_name.set(self.obs.meteor_name)
        self.scale_progress.config(to=self.obs.frame_count - 1)
        self.spin_frame.config(to=self.obs.frame_count - 1)

        self.canvas_editor.mask_data = self.obs.mask_frame
        self.canvas_editor.signal_data = self.obs.signal_frame

        self.check_status()
        self.update_video_frame()
        self.canvas_editor.fit_to_window()

    def apply_time(self):
        try:
            y, m, d = self.var_y.get(), self.var_m.get(), self.var_d.get()
            hh, mm, ss = self.var_hh.get(), self.var_mm.get(), self.var_ss.get()

            dt_base = datetime.strptime(f"{y}-{m}-{d} {hh}:{mm}", "%Y-%m-%d %H:%M")
            dt_base += timedelta(seconds=float(ss))

            ref_idx = self.var_cal_frame.get()
            indices = np.arange(self.obs.frame_count)

            abs_times = dt_base.timestamp() + (indices - ref_idx) * (1.0 / self.eff_fps)

            self.obs.mean_time = np.mean(abs_times)
            self.obs.frame_time = abs_times - self.obs.mean_time
            self.obs.frame_exposure = np.full(self.obs.frame_count, 1.0 / self.eff_fps)

            if "time" not in self.obs.contents:
                self.obs.contents.append("time")

            self.check_status()
            self.update_time_display(self.curr_frame_idx)

        except Exception as e:
            messagebox.showerror("错误", f"时间格式错误或计算失败: {str(e)}")

    def on_video_seek(self, val):
        self.curr_frame_idx = int(float(val))
        self.var_cal_frame.set(self.curr_frame_idx)
        self.update_video_frame()

    def on_spin_seek(self):
        self.curr_frame_idx = self.var_cal_frame.get()
        self.scale_progress.set(self.curr_frame_idx)
        self.update_video_frame()

    def on_frame_scroll(self, direction):
        if self.obs.data is None:
            return
        self.stop_play()
        new_idx = self.curr_frame_idx + direction
        if 0 <= new_idx < self.obs.frame_count:
            self.curr_frame_idx = new_idx
            self.var_cal_frame.set(self.curr_frame_idx)
            self.scale_progress.set(self.curr_frame_idx)
            self.update_video_frame()

    def update_time_display(self, idx):
        if "time" in self.obs.contents and self.obs.mean_time:
            ts = self.obs.mean_time + self.obs.frame_time[idx]
            dt = datetime.fromtimestamp(ts)
            self.var_y.set(dt.year)
            self.var_m.set(f"{dt.month:02d}")
            self.var_d.set(f"{dt.day:02d}")
            self.var_hh.set(f"{dt.hour:02d}")
            self.var_mm.set(f"{dt.minute:02d}")
            self.var_ss.set(f"{dt.second + dt.microsecond/1e6:.3f}")

    def update_video_frame(self):
        if self.obs.data is None:
            return
        self.lbl_frame_info.config(
            text=f"{self.curr_frame_idx}/{self.obs.frame_count-1}"
        )

        if self.mode_var.get() in ["mask", "signal"]:
            if self.obs.max_frame is not None:
                self.canvas_editor.set_image(self.obs.max_frame)
            else:
                self.canvas_editor.set_image(self.obs.data[0])
        elif self.mode_var.get() == "star":
            if self.obs.mean_frame is not None:
                self.canvas_editor.set_image(self.obs.mean_frame.astype(np.uint8))
            else:
                self.canvas_editor.set_image(self.obs.data[0])
        else:
            self.canvas_editor.set_image(self.obs.data[self.curr_frame_idx])

        self.update_time_display(self.curr_frame_idx)

    def toggle_play(self):
        if not self.is_playing:
            self.is_playing = True
            self.play_loop()

    def stop_play(self):
        self.is_playing = False

    def play_loop(self):
        if not self.is_playing:
            return

        self.curr_frame_idx += 1
        if self.curr_frame_idx >= self.obs.frame_count:
            self.curr_frame_idx = 0

        self.scale_progress.set(self.curr_frame_idx)
        self.var_cal_frame.set(self.curr_frame_idx)
        self.update_video_frame()

        delay = int(1000 / self.eff_fps)
        self.root.after(delay, self.play_loop)

    def on_mode_change(self):
        m = self.mode_var.get()
        self.canvas_editor.mode = m

        if m == "mask":
            self.flag_mask_edited = True
        elif m == "signal":
            self.flag_signal_edited = True

        self.check_status()
        if m in ["mask", "signal"] and self.obs.max_frame is not None:
            self.canvas_editor.set_image(self.obs.max_frame)
        elif m == "star" and self.obs.mean_frame is not None:
            self.canvas_editor.set_image(self.obs.mean_frame.astype(np.uint8))
        else:
            self.update_video_frame()

    def on_canvas_edited(self):
        self.check_status()

    def clear_layer(self):
        m = self.canvas_editor.mode
        if m == "mask":
            self.obs.mask_frame[:] = 1
        elif m == "signal":
            self.obs.signal_frame[:] = 0
        self.canvas_editor.redraw()

    def save_mask_file(self):
        if self.obs.mask_frame is None:
            return
        f = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG", "*.png")]
        )
        if f:
            cv2.imwrite(f, self.obs.mask_frame * 255)
            messagebox.showinfo("信息", "遮罩已保存")

    def load_mask_file(self):
        f = filedialog.askopenfilename(filetypes=[("PNG", "*.png")])
        if f and self.obs.data is not None:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape[:2] != (self.obs.image_height, self.obs.image_width):
                    img = cv2.resize(img, (self.obs.image_width, self.obs.image_height))

                _, thresh = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
                self.obs.mask_frame = thresh.astype(np.uint8)
                self.canvas_editor.mask_data = self.obs.mask_frame
                self.flag_mask_edited = True
                self.canvas_editor.redraw()
                self.check_status()

    def add_star_point(self, x, y):
        self.canvas_editor.star_list.append(
            {
                "id": chr(ord("a") + len(self.canvas_editor.star_list)),
                "x": x,
                "y": y,
                "name": "",
            }
        )
        self.canvas_editor.redraw()
        if self.star_win and self.star_win.winfo_exists():
            self.star_win.refresh_table()

    def validate_star_input(self, input_val):
        """验证输入的名称或ID是否有效，返回 (True, hip_id) 或 (False, ErrorMsg)"""
        val = input_val.strip()
        if not val:
            return False, "名称不能为空"

        hip_id = None

        # 1. 解析 HIP ID
        if val.isdigit():
            hip_id = int(val)
        elif val in self.star_names_map:
            hip_id = self.star_names_map[val]
        else:
            return False, "未找到恒星中文名或英文名。\n请使用HIP编号或有效名称。"

        # 2. 验证是否在星表中 (仅当 Skyfield 可用且数据已加载时)
        if HAS_SKYFIELD and self.hipparcos_df is not None:
            if hip_id not in self.hipparcos_df.index:
                return (
                    False,
                    "恒星编号不存在。\n请确认编号正确或星表文件完整。",
                )
        elif not HAS_SKYFIELD:
            # 如果没有 skyfield，我们无法验证 ID 是否真实存在，只能姑且认为是有效的
            pass

        return True, hip_id

    def open_star_table(self):
        if self.star_win and self.star_win.winfo_exists():
            self.star_win.lift()
            return
        # 传递 validate_star_input 作为新的回调参数
        self.star_win = StarDialog(
            self.root,
            self.canvas_editor.star_list,
            self.on_star_update,
            self.star_names_map,
            self.star_id_to_name,
            self.validate_star_input,
        )

    def on_star_update(self):
        # 计算 ECI 坐标
        if HAS_SKYFIELD and self.hipparcos_df is not None and self.obs.mean_time:
            t = self.ts.from_astropy(Time(self.obs.mean_time, format="unix"))
            for s in self.canvas_editor.star_list:
                input_name = s.get("name", "").strip()
                if not input_name:
                    continue

                hip_id = None
                # 判断是数字(HIP)还是名字
                if input_name.isdigit():
                    hip_id = int(input_name)
                elif input_name in self.star_names_map:
                    hip_id = self.star_names_map[input_name]

                if hip_id:
                    try:
                        star = Star.from_dataframe(self.hipparcos_df.loc[hip_id])
                        pos = self.earth.at(t).observe(star).position.au
                        # 归一化 ECI
                        mag = np.linalg.norm(pos)
                        s["eci_x"], s["eci_y"], s["eci_z"] = pos / mag
                    except Exception as e:
                        # 可能是HIP编号超出范围或不存在
                        print(f"Star ECI error (HIP {hip_id}): {e}")

        self.canvas_editor.redraw()
        self.check_status()

    def check_status(self):
        c_time = "time" in self.obs.contents
        c_image = self.flag_mask_edited and self.flag_signal_edited
        valid_stars = [s for s in self.canvas_editor.star_list if s.get("name")]
        c_star = len(valid_stars) >= 5

        self.lbl_s_time.config(bg="#90EE90" if c_time else "#FFB6C1")
        self.lbl_s_image.config(bg="#90EE90" if c_image else "#FFB6C1")
        self.lbl_s_star.config(bg="#90EE90" if c_star else "#FFB6C1")

        if c_time and c_image:
            self.btn_save_fits.config(state="normal")
            self.btn_save_files.config(state="normal")
        else:
            self.btn_save_fits.config(state="disabled")
            self.btn_save_files.config(state="disabled")

    def _prepare_obs_data(self):
        self.obs.meteor_name = self.var_m_name.get()
        self.obs.station_name = self.var_s_name.get()
        self.obs.camera_name = self.var_c_name.get()

        try:
            lat = float(self.var_lat.get())
            lon = float(self.var_lon.get())
            alt = float(self.var_alt.get())
            self.obs.station_location = (lat, lon, alt)
        except Exception:
            pass

        if "image" not in self.obs.contents:
            self.obs.contents.append("image")

        valid_stars = [s for s in self.canvas_editor.star_list if s.get("name")]

        if valid_stars:
            n = len(valid_stars)
            self.obs.star_name = []
            self.obs.star_pixel_coord = np.zeros((2, n))
            self.obs.star_eci_coord = np.zeros((3, n))

            for i, s in enumerate(valid_stars):
                input_name = s.get("name", "").strip()
                hip_id = 0

                # 最终保存时再次解析是数字还是名字
                if input_name.isdigit():
                    hip_id = int(input_name)
                else:
                    hip_id = self.star_names_map.get(input_name, 0)

                self.obs.star_name.append(hip_id)
                self.obs.star_pixel_coord[:, i] = [s["x"], s["y"]]
                if "eci_x" in s:
                    self.obs.star_eci_coord[:, i] = [s["eci_x"], s["eci_y"], s["eci_z"]]

            self.obs.star_method = "manual_gui"
            if "star" not in self.obs.contents:
                self.obs.contents.append("star")
        else:
            if "star" in self.obs.contents:
                self.obs.contents.remove("star")

        if self.obs.objects is None:
            self.obs.objects = []

    def save_fits(self):
        self._prepare_obs_data()
        f = filedialog.asksaveasfilename(
            defaultextension=".fits", filetypes=[("FITS", "*.fits")]
        )
        if f:
            try:
                self.obs.to_fits(f)
                messagebox.showinfo("成功", "FITS 保存成功")
            except Exception as e:
                messagebox.showerror("失败", str(e))

    def save_files(self):
        self._prepare_obs_data()
        f = filedialog.asksaveasfilename(title="保存基础文件名")
        if f:
            try:
                self.obs.to_files(f)
                messagebox.showinfo("成功", "保存成功")
            except Exception as e:
                messagebox.showerror("失败", str(e))


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass

    app = ConverterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
