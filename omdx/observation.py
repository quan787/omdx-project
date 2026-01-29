# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import json
import numpy as np
from astropy.io import fits
from astropy.table import Table
import cv2
from typing import Optional, List, Dict, Any, Union, Tuple, Type, TypeVar
from ._version import __version__

# 定义泛型变量，用于类方法的返回类型注解
T = TypeVar("T", bound="MeteorObservation")

VERSION = __version__


class NumpyEncoder(json.JSONEncoder):
    """
    用于处理 Numpy 数据类型的 JSON 编码器。
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


class ObservationObject:
    """
    存储单个流星目标观测数据的容器类。
    """

    def __init__(self) -> None:
        # 流星在每一帧中的索引列表
        self.meteor_index: Optional[List[int]] = None
        # 流星的像素坐标，形状通常为 (2, N)
        self.meteor_pixel_coord: Optional[np.ndarray] = None
        # 流星的 ECI（地心惯性）坐标，形状通常为 (3, N)
        self.meteor_eci_coord: Optional[np.ndarray] = None
        # 目标类型（如流星、卫星等）
        self.object_type: Optional[str] = None
        # 流星的光通量，长度为 N
        self.meteor_flux: Optional[np.ndarray] = None
        # 流星的星等，长度为 N
        self.meteor_magnitude: Optional[np.ndarray] = None


class MeteorObservation:
    """
    流星观测数据的主类，包含图像数据、时间信息、天体测量、测光及校准数据。
    支持 FITS、JSON、MP4 等格式的导入导出。
    """

    def __init__(self) -> None:
        # --- 版本信息 ---
        self.version: str = VERSION

        # --- 基础信息 ---
        self.station_name: Optional[str] = None  # 站点名称
        self.camera_name: Optional[str] = None  # 相机名称
        self.meteor_name: Optional[str] = None  # 流星名称
        self.mode: Optional[str] = None  # 观测模式，如 fast，slow
        # 站点经纬高 (lat, lon, alt)
        self.station_location: Optional[
            Union[List[float], Tuple[float, float, float]]
        ] = None
        # 相机颜色模式：单色为A，彩色为RGB，BGR，RGGB，GRBG等等
        self.color: Optional[str] = None

        # --- 阶段/内容 ---
        # 包含的选项：image, time, star, meteor, calibration, photometry, database
        self.contents: List[str] = []
        self.processed_time: Optional[Union[float, str]] = (
            None  # 处理时间，unix时间戳或字符串
        )

        # --- Image (图像数据) ---
        self.image_width: Optional[int] = None
        self.image_height: Optional[int] = None
        self.frame_count: Optional[int] = None
        # 图像数据，shape=(frame_count, height, width) 或 (frame_count, height, width, channels)
        self.data: Optional[np.ndarray] = None
        self._mean_frame: Optional[np.ndarray] = None
        # 流星的轮廓/信号帧，流星区域是1，其他是0
        self.signal_frame: Optional[np.ndarray] = None
        self._std_frame: Optional[np.ndarray] = None
        self._max_frame: Optional[np.ndarray] = None
        # 图像遮罩，探测区域是1，其他是0
        self.mask_frame: Optional[np.ndarray] = None

        # --- Time (时间信息) ---
        self.time_method: Optional[str] = None
        self.time_source: Optional[str] = None  # 时间来源，camera, system, gps, ntp
        self.mean_time: Optional[float] = None  # 流星平均时间，unix时间戳
        # 每帧时间与平均时间的插值，shape=(frame_count,)
        self.frame_time: Optional[np.ndarray] = None
        # 每帧曝光时间，单位秒，shape=(frame_count,)
        self.frame_exposure: Optional[np.ndarray] = None

        # --- Star (恒星信息) ---
        self.star_method: Optional[str] = None
        # 恒星像素坐标，shape=(2, N)
        self.star_pixel_coord: Optional[np.ndarray] = None
        # 恒星 ECI 坐标，shape=(3, N)
        self.star_eci_coord: Optional[np.ndarray] = None
        # 恒星名称（HIP星表编号），列表，长度N
        self.star_name: Optional[List[str]] = None
        self.star_magnitude: Optional[np.ndarray] = None
        self.star_flux: Optional[np.ndarray] = None

        # --- Meteor (流星对象) ---
        self.object_method: Optional[str] = None
        self.objects: Optional[List[ObservationObject]] = None

        # --- Calibration (校准/定标) ---
        self.calibration_method: Optional[str] = None
        self.calibration_result: Optional[str] = None
        # 内部存储校准模型参数的字典
        self._calibration_model: Optional[Dict[str, Union[str, int, float]]] = None
        self.calibration_residual: Optional[float] = None

        # --- Photometry (测光) ---
        self.photometry_method: Optional[str] = None
        # 内部存储测光模型参数的字典
        self._photometry_model: Optional[Dict[str, Union[str, int, float]]] = None
        self.photometry_residual: Optional[float] = None

        # --- Database (数据库ID) ---
        self.station_id: Optional[Union[str, int]] = None  # 数据库中站点的唯一ID
        self.meteor_id: Optional[Union[str, int]] = None  # 数据库中流星的唯一ID

    @property
    def max_frame(self) -> Optional[np.ndarray]:
        """
        获取最大值合成帧。如果未缓存，则根据 data 计算。
        """
        if self._max_frame is None:
            if self.data is not None and len(self.data) > 0:
                maxmatric = self.data[0]
                for i in range(1, len(self.data)):
                    maxmatric = np.maximum(maxmatric, self.data[i])
                self._max_frame = np.array(maxmatric, dtype=self.data.dtype)
            else:
                if "image" not in self.contents:
                    return None
                raise ValueError("Data is missing but image content is declared.")
        return self._max_frame

    @max_frame.setter
    def max_frame(self, array_in: np.ndarray) -> None:
        self._max_frame = array_in

    @property
    def mean_frame(self) -> Optional[np.ndarray]:
        """
        获取平均值合成帧。如果未缓存，则根据 data 计算。
        """
        if self._mean_frame is None:
            if self.data is not None and len(self.data) > 0:
                # 显式转换为 float 以防溢出
                stack = np.sum(np.array(self.data, dtype=float), axis=0)
                if self.max_frame is None:
                    raise ValueError("Cannot calculate mean_frame without max_frame.")
                # 注意：此处原逻辑似乎是 (sum - max) / (N-1)，即去除最大值后的平均
                self._mean_frame = (stack - self.max_frame) / (len(self.data) - 1)
            else:
                if "image" not in self.contents:
                    return None
                raise ValueError("Data is missing but image content is declared.")
        return self._mean_frame

    @mean_frame.setter
    def mean_frame(self, array_in: np.ndarray) -> None:
        self._mean_frame = array_in

    @property
    def std_frame(self) -> Optional[np.ndarray]:
        """
        获取标准差帧。如果未缓存，则根据 data 计算。
        """
        if self._std_frame is None:
            if self.data is not None and len(self.data) > 0:
                meanframe = self.mean_frame
                if meanframe is None or self.max_frame is None:
                    raise ValueError(
                        "Cannot calculate std_frame without mean and max frames."
                    )

                # 初始化 stack: -(max - mean)^2
                stack = -np.power((self.max_frame - meanframe), 2)
                for i in range(len(self.data)):
                    stack = stack + np.power(
                        (np.array(self.data[i], dtype=float) - meanframe), 2
                    )
                self._std_frame = np.sqrt(stack / (len(self.data) - 1))
            else:
                if "image" not in self.contents:
                    return None
                raise ValueError("Data is missing but image content is declared.")
        return self._std_frame

    @std_frame.setter
    def std_frame(self, array_in: np.ndarray) -> None:
        self._std_frame = array_in

    def _validate_flat_dict(self, value: Dict[str, Any], attr_name: str) -> None:
        """
        校验字典格式是否为扁平结构（Key为短字符串，Value为数字或字符串）。

        Args:
            value: 待校验的字典。
            attr_name: 属性名称，用于错误提示。

        Raises:
            ValueError: 当字典格式不符合 FITS Header 兼容性要求时抛出。
        """
        if not isinstance(value, dict):
            raise ValueError(f"{attr_name} 必须是一个字典。")

        for k, v in value.items():
            # 1. 校验 Key 的长度和字符
            if not isinstance(k, str):
                raise ValueError(f"{attr_name} 的 key '{k}' 必须是字符串。")

            if len(k) > 4:
                raise ValueError(f"{attr_name} 的 key '{k}' 超过了4个字符限制。")

            # 校验 key 是否只包含数字、小写字母、- 和 _
            if not all(c.isdigit() or c.islower() or c in "-_" for c in k):
                raise ValueError(
                    f"{attr_name} 的 key '{k}' 包含非法字符（仅限小写字母、数字、-、_）。"
                )

            # 2. 校验 Value 是否为扁平的字符串或数字
            if not isinstance(v, (str, int, float)):
                raise ValueError(
                    f"{attr_name} 的值 '{v}' 类型错误（仅限字符串或数字，不可嵌套）。"
                )

    @property
    def calibration_model(self) -> Optional[Dict[str, Union[str, int, float]]]:
        return self._calibration_model

    @calibration_model.setter
    def calibration_model(
        self, value: Optional[Dict[str, Union[str, int, float]]]
    ) -> None:
        if value is not None:
            self._validate_flat_dict(value, "calibration_model")
        self._calibration_model = value

    @property
    def photometry_model(self) -> Optional[Dict[str, Union[str, int, float]]]:
        return self._photometry_model

    @photometry_model.setter
    def photometry_model(
        self, value: Optional[Dict[str, Union[str, int, float]]]
    ) -> None:
        if value is not None:
            self._validate_flat_dict(value, "photometry_model")
        self._photometry_model = value

    def to_fits(self, fits_path: str = "observation.fits") -> None:
        """
        将观测数据保存为 FITS 文件格式。

        Args:
            fits_path: 输出 FITS 文件的路径。
        """
        # --- 1. 创建 Primary HDU (基础信息、处理方法与合成图) ---
        # 合成图逻辑：结合 mask_frame 和 signal_frame
        if self.image_height is None or self.image_width is None:
            # 如果没有尺寸信息，设为 0 (通常应避免这种情况)
            h, w = 0, 0
        else:
            h, w = self.image_height, self.image_width

        mixed_data = np.zeros((h, w), dtype=np.uint8)
        if self.mask_frame is not None:
            mixed_data += (self.mask_frame > 0).astype(np.uint8) * 64
        if self.signal_frame is not None:
            mixed_data += (self.signal_frame > 0).astype(np.uint8) * 128

        primary_hdu: fits.PrimaryHDU
        if "image" in self.contents:
            primary_hdu = fits.PrimaryHDU(mixed_data)
        else:
            primary_hdu = fits.PrimaryHDU()
        hdr = primary_hdu.header

        # 基础元数据
        hdr["M_VER"] = self.version
        hdr["M_STA"] = self.station_name
        hdr["M_CAM"] = self.camera_name
        hdr["M_NAME"] = self.meteor_name
        hdr["M_MEANT"] = self.mean_time
        hdr["M_TSRC"] = self.time_source
        hdr["M_MODE"] = self.mode
        if self.station_location:
            hdr["M_STALAT"], hdr["M_STALON"], hdr["M_STAALT"] = self.station_location
        hdr["M_COLOR"] = self.color
        hdr["M_W"] = w
        hdr["M_H"] = h
        hdr["M_FCNT"] = self.frame_count
        hdr["M_CONTS"] = ",".join(self.contents)
        hdr["M_PROCT"] = self.processed_time if self.processed_time else ""

        # 各种处理方法名称
        hdr["M_T_MTH"] = self.time_method

        if "star" in self.contents:
            hdr["M_S_MTH"] = self.star_method
        if "meteor" in self.contents:
            hdr["M_O_MTH"] = self.object_method
            if self.objects:
                hdr["M_O_CNT"] = len(self.objects)
                for obj_idx, obj in enumerate(self.objects):
                    if obj.object_type is not None:
                        hdr[f"M_O_TP{obj_idx:02X}"] = obj.object_type
            else:
                hdr["M_O_CNT"] = 0

        if "calibration" in self.contents:
            hdr["M_C_MTH"] = self.calibration_method
        if "photometry" in self.contents:
            hdr["M_P_MTH"] = self.photometry_method
        if "database" in self.contents:
            hdr["M_STAID"] = self.station_id if self.station_id else ""
            hdr["M_METID"] = self.meteor_id if self.meteor_id else ""

        hdul = fits.HDUList([primary_hdu])

        # --- 2. 建立流星数据的帧索引映射 ---
        # 格式: {frame_index: [(obj_list_index, internal_array_index), ...]}
        obj_lookup: Dict[int, List[Tuple[int, int]]] = {}
        if self.objects:
            for obj_idx, obj in enumerate(self.objects):
                if obj.meteor_index is not None:
                    for internal_idx, frame_idx in enumerate(obj.meteor_index):
                        if frame_idx not in obj_lookup:
                            obj_lookup[frame_idx] = []
                        obj_lookup[frame_idx].append((obj_idx, internal_idx))

        # --- 3. 写入每一帧图像 HDU (含每帧流星坐标与测光) ---
        count = self.frame_count if self.frame_count is not None else 0
        for i in range(count):
            if "image" in self.contents:
                if self.data is not None and i < len(self.data):
                    frame_data = self.data[i]
                else:
                    frame_data = np.zeros((h, w))
                image_hdu = fits.ImageHDU(frame_data, name=f"M_FRAME_{i:05d}")
            else:
                image_hdu = fits.ImageHDU(name=f"M_FRAME_{i:05d}")
            ihdr = image_hdu.header

            # 时间信息
            if self.frame_time is not None and i < len(self.frame_time):
                ihdr["M_FTIME"] = self.frame_time[i]
            else:
                ihdr["M_FTIME"] = 0

            if self.frame_exposure is not None and i < len(self.frame_exposure):
                ihdr["M_EXPOS"] = self.frame_exposure[i]

            # 写入该帧包含的所有流星信息 (使用十六进制索引)
            if i in obj_lookup and self.objects:
                for obj_idx, int_idx in obj_lookup[i]:
                    hex_id = f"{obj_idx:02X}"
                    obj = self.objects[obj_idx]

                    # 像素坐标
                    if "meteor" in self.contents and obj.meteor_pixel_coord is not None:
                        ihdr[f"M_O_PX{hex_id}"] = obj.meteor_pixel_coord[0][int_idx]
                        ihdr[f"M_O_PY{hex_id}"] = obj.meteor_pixel_coord[1][int_idx]

                    # ECI 坐标
                    if (
                        "meteor" in self.contents
                        and "calibration" in self.contents
                        and obj.meteor_eci_coord is not None
                    ):
                        ihdr[f"M_O_EX{hex_id}"] = obj.meteor_eci_coord[0][int_idx]
                        ihdr[f"M_O_EY{hex_id}"] = obj.meteor_eci_coord[1][int_idx]
                        ihdr[f"M_O_EZ{hex_id}"] = obj.meteor_eci_coord[2][int_idx]

                    # 测光结果 (Flux 与 Magnitude)
                    if "meteor" in self.contents and "photometry" in self.contents:
                        if obj.meteor_flux is not None:
                            ihdr[f"M_O_FX{hex_id}"] = obj.meteor_flux[int_idx]
                        if obj.meteor_magnitude is not None:
                            ihdr[f"M_O_MG{hex_id}"] = obj.meteor_magnitude[int_idx]
            hdul.append(image_hdu)

        # --- 4. 写入星表与模型参数 (BinTable HDU) ---
        if "star" in self.contents:
            star_table_data: Dict[str, Any] = {
                "star_name": self.star_name if self.star_name is not None else [],
                "star_pic_x": (
                    self.star_pixel_coord[0]
                    if self.star_pixel_coord is not None
                    else []
                ),
                "star_pic_y": (
                    self.star_pixel_coord[1]
                    if self.star_pixel_coord is not None
                    else []
                ),
                "star_eci_x": (
                    self.star_eci_coord[0] if self.star_eci_coord is not None else []
                ),
                "star_eci_y": (
                    self.star_eci_coord[1] if self.star_eci_coord is not None else []
                ),
                "star_eci_z": (
                    self.star_eci_coord[2] if self.star_eci_coord is not None else []
                ),
            }
            if self.star_magnitude is not None:
                star_table_data["star_mag"] = self.star_magnitude
            if self.star_flux is not None:
                star_table_data["star_flux"] = self.star_flux

            # 确定表格行数并创建 Table
            if (
                star_table_data["star_name"] is not None
                and len(star_table_data["star_name"]) > 0
            ):
                t = Table(star_table_data)
            else:
                t = Table(names=list(star_table_data.keys()))

            table_hdu = fits.BinTableHDU(t, name="M_STAR")
            thdr = table_hdu.header

            # 校准与测光残差
            thdr["M_CRSLT"] = self.calibration_result if self.calibration_result else ""
            thdr["M_CRES"] = (
                self.calibration_residual if self.calibration_residual else -1
            )
            thdr["M_PRES"] = (
                self.photometry_residual if self.photometry_residual else -1
            )

            # 写入校准模型字典参数 (M_C_ 开头)
            if isinstance(self.calibration_model, dict):
                for k, v in self.calibration_model.items():
                    thdr[f"M_C_{str(k)[:4].upper()}"] = str(v)

            # 写入测光模型字典参数 (M_P_ 开头)
            if isinstance(self.photometry_model, dict):
                for k, v in self.photometry_model.items():
                    thdr[f"M_P_{str(k)[:4].upper()}"] = str(v)

            hdul.append(table_hdu)

        # 保存文件
        hdul.writeto(fits_path, overwrite=True)

    @classmethod
    def from_fits(cls: Type[T], fits_path: str = "observation.fits") -> T:
        """
        从 FITS 文件中读取数据并恢复 MeteorObservation 实例。

        Args:
            fits_path: FITS 文件路径。

        Returns:
            MeteorObservation: 恢复的观测对象。
        """
        obs = cls()
        with fits.open(fits_path) as hdul:
            # --- 1. 读取 Primary HDU (基础信息) ---
            primary_hdu = hdul[0]
            hdr = primary_hdu.header

            # 基础属性
            obs.version = str(hdr.get("M_VER", VERSION))
            obs.station_name = hdr.get("M_STA")
            obs.camera_name = hdr.get("M_CAM")
            obs.meteor_name = hdr.get("M_NAME")
            obs.mean_time = hdr.get("M_MEANT")
            obs.time_source = hdr.get("M_TSRC")
            obs.mode = hdr.get("M_MODE")
            obs.color = hdr.get("M_COLOR")
            obs.image_width = hdr.get("M_W")
            obs.image_height = hdr.get("M_H")
            obs.frame_count = hdr.get("M_FCNT")
            obs.processed_time = hdr.get("M_PROCT")

            # 恢复 location tuple
            if "M_STALAT" in hdr:
                obs.station_location = [
                    float(hdr["M_STALAT"]),
                    float(hdr["M_STALON"]),
                    float(hdr["M_STAALT"]),
                ]

            # 恢复 contents 列表
            conts_str = str(hdr.get("M_CONTS", ""))
            obs.contents = conts_str.split(",") if conts_str else []

            # 恢复 Method
            obs.time_method = hdr.get("M_T_MTH")
            if "star" in obs.contents:
                obs.star_method = hdr.get("M_S_MTH")

            # 读取流星类型列表
            obj_types: List[Optional[str]] = []
            if "meteor" in obs.contents:
                obs.object_method = hdr.get("M_O_MTH")
                obj_cnt = hdr.get("M_O_CNT", 0)
                obj_types = [hdr.get(f"M_O_TP{idx:02X}") for idx in range(obj_cnt)]

            if "calibration" in obs.contents:
                obs.calibration_method = hdr.get("M_C_MTH")
            if "photometry" in obs.contents:
                obs.photometry_method = hdr.get("M_P_MTH")
            if "database" in obs.contents:
                obs.station_id = hdr.get("M_STAID")
                obs.meteor_id = hdr.get("M_METID")

            # 恢复合成图 (Mask 和 Signal)
            # 写入逻辑: mask * 64 + signal * 128
            if "image" in obs.contents:
                mixed_data = primary_hdu.data
                if mixed_data is not None:
                    # 提取 Mask (检查 Bit 6)
                    obs.mask_frame = (mixed_data & 64) > 0
                    # 提取 Signal (检查 Bit 7)
                    obs.signal_frame = (mixed_data & 128) > 0

            # --- 2. 读取 Image HDUs (序列帧与流星数据) ---
            obj_count = hdr.get("M_O_CNT", 0)
            # 使用临时字典存储分散在各帧的流星数据
            temp_objects: Dict[int, Dict[str, List[Any]]] = {
                i: {
                    "inds": [],
                    "px": [],
                    "py": [],
                    "ex": [],
                    "ey": [],
                    "ez": [],
                    "fx": [],
                    "mg": [],
                }
                for i in range(obj_count)
            }

            frames_data = []
            frames_time = []
            frames_exposure = []

            # 遍历所有 HDU 查找 M_FRAME_
            frame_hdus: List[Tuple[int, Any]] = []
            for hdu in hdul:
                if hdu.name.startswith("M_FRAME_"):
                    try:
                        idx = int(hdu.name.split("_")[-1])
                        frame_hdus.append((idx, hdu))
                    except ValueError:
                        continue

            # 按帧序号排序
            frame_hdus.sort(key=lambda x: x[0])

            for idx, hdu in frame_hdus:
                ihdr = hdu.header
                # 图像数据
                frames_data.append(hdu.data)

                # 时间信息
                if "time" in obs.contents:
                    frames_time.append(ihdr.get("M_FTIME", 0))
                    frames_exposure.append(ihdr.get("M_EXPOS", 0))

                # 读取该帧中的流星数据
                if "meteor" in obs.contents:
                    for obj_i in range(obj_count):
                        hex_id = f"{obj_i:02X}"
                        # 检查该帧是否有该对象的记录 (以 M_PX 为标志)
                        if f"M_O_PX{hex_id}" in ihdr:
                            temp_objects[obj_i]["inds"].append(idx)
                            temp_objects[obj_i]["px"].append(ihdr[f"M_O_PX{hex_id}"])
                            temp_objects[obj_i]["py"].append(ihdr[f"M_O_PY{hex_id}"])

                            if "calibration" in obs.contents:
                                temp_objects[obj_i]["ex"].append(
                                    ihdr.get(f"M_O_EX{hex_id}", 0)
                                )
                                temp_objects[obj_i]["ey"].append(
                                    ihdr.get(f"M_O_EY{hex_id}", 0)
                                )
                                temp_objects[obj_i]["ez"].append(
                                    ihdr.get(f"M_O_EZ{hex_id}", 0)
                                )

                            if "photometry" in obs.contents:
                                temp_objects[obj_i]["fx"].append(
                                    ihdr.get(f"M_O_FX{hex_id}", 0)
                                )
                                temp_objects[obj_i]["mg"].append(
                                    ihdr.get(f"M_O_MG{hex_id}", 0)
                                )

            # 设置图像相关属性
            obs.frame_time = np.array(frames_time)
            obs.frame_exposure = np.array(frames_exposure)
            if "image" in obs.contents:
                obs.data = np.array(frames_data)

            # 重组流星对象
            if "meteor" in obs.contents:
                obs.objects = []
                for i in range(obj_count):
                    obj_data = temp_objects[i]
                    if not obj_data["inds"]:
                        continue  # 空对象跳过

                    met = ObservationObject()
                    met.object_type = obj_types[i]
                    met.meteor_index = obj_data["inds"]

                    # 像素坐标重组 (2, N)
                    met.meteor_pixel_coord = np.array([obj_data["px"], obj_data["py"]])

                    # ECI 坐标重组 (3, N)
                    if "calibration" in obs.contents and obj_data["ex"]:
                        met.meteor_eci_coord = np.array(
                            [obj_data["ex"], obj_data["ey"], obj_data["ez"]]
                        )

                    # 测光重组
                    if "photometry" in obs.contents:
                        if obj_data["fx"]:
                            met.meteor_flux = np.array(obj_data["fx"])
                        if obj_data["mg"]:
                            met.meteor_magnitude = np.array(obj_data["mg"])

                    obs.objects.append(met)

            # --- 3. 读取 BinTable HDU (星表与模型) ---
            if "star" in obs.contents and "M_STAR" in hdul:
                table_hdu = hdul["M_STAR"]
                thdr = table_hdu.header
                tdata = table_hdu.data

                # 恢复星表数据
                obs.star_name = (
                    tdata["star_name"].tolist() if "star_name" in tdata.names else []
                )

                px = tdata["star_pic_x"] if "star_pic_x" in tdata.names else []
                py = tdata["star_pic_y"] if "star_pic_y" in tdata.names else []
                if len(px) > 0:
                    obs.star_pixel_coord = np.array([px, py])

                ex = tdata["star_eci_x"] if "star_eci_x" in tdata.names else []
                ey = tdata["star_eci_y"] if "star_eci_y" in tdata.names else []
                ez = tdata["star_eci_z"] if "star_eci_z" in tdata.names else []
                if len(ex) > 0:
                    obs.star_eci_coord = np.array([ex, ey, ez])

                if "star_mag" in tdata.names:
                    obs.star_magnitude = tdata["star_mag"]
                if "star_flux" in tdata.names:
                    obs.star_flux = tdata["star_flux"]

                # 恢复校准结果
                obs.calibration_result = thdr.get("M_CRSLT")
                obs.calibration_residual = thdr.get("M_CRES")
                if obs.calibration_residual == -1:
                    obs.calibration_residual = None

                obs.photometry_residual = thdr.get("M_PRES")
                if obs.photometry_residual == -1:
                    obs.photometry_residual = None

                # 恢复模型字典参数
                c_model: Dict[str, Union[int, float]] = {}
                p_model: Dict[str, Union[int, float]] = {}
                for key in thdr.keys():
                    if key.startswith("M_C_"):
                        # 去掉前缀，转小写
                        real_key = key[4:].lower()
                        # 值是字符串，尝试转数字
                        val = thdr[key]
                        try:
                            if "." in str(val):
                                val_num = float(val)
                            else:
                                val_num = int(val)
                            c_model[real_key] = val_num
                        except ValueError:
                            pass  # 保持字符串或忽略

                    elif key.startswith("M_P_"):
                        real_key = key[4:].lower()
                        val = thdr[key]
                        try:
                            if "." in str(val):
                                val_num = float(val)
                            else:
                                val_num = int(val)
                            p_model[real_key] = val_num
                        except ValueError:
                            pass

                if c_model:
                    obs.calibration_model = c_model  # type: ignore
                if p_model:
                    obs.photometry_model = p_model  # type: ignore

        return obs

    def to_files(self, base_path: Optional[str] = None) -> None:
        """
        保存为 JSON (元数据), MP4 (视频), PNG (Mask/Signal)。

        Args:
            base_path: 基础路径或文件名 (不含扩展名或含任意扩展名均可)。
                       如果为 None，则尝试使用 meteor_name。
        """
        if base_path is None:
            base_path = self.meteor_name if self.meteor_name else "meteor_obs"

        # 去除扩展名，获取基础文件名
        file_root = os.path.splitext(base_path)[0]
        json_path = file_root + ".json"
        mp4_path = file_root + ".mp4"
        png_path = file_root + ".png"

        # --- 1. 保存 JSON (元数据) ---
        # 提取字典，过滤 image 相关的大数据字段
        meta_dict: Dict[str, Any] = {}
        skip_keys = [
            "data",
            "_mean_frame",
            "_std_frame",
            "_max_frame",
            "signal_frame",
            "mask_frame",
            "objects",
        ]

        for k, v in self.__dict__.items():
            if k not in skip_keys and v is not None:
                clean_key = k.lstrip("_")
                meta_dict[clean_key] = v

        bayer_list = ["RGGB", "BGGR", "GRBG", "GBRG"]
        if self.color in bayer_list or self.color == "RGB":
            meta_dict["color"] = "BGR"  # 强制 JSON 中的记录与视频转换后的 BGR 一致
        elif self.color is None or self.color == "A":
            meta_dict["color"] = "A"

        # 特殊处理 objects 列表
        if self.objects is not None:
            objs_list = []
            for obj in self.objects:
                o_dict = {k: v for k, v in obj.__dict__.items() if v is not None}
                objs_list.append(o_dict)
            meta_dict["objects"] = objs_list

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, cls=NumpyEncoder, indent=4, ensure_ascii=False)

        # --- 2. 保存 MP4 (视频) ---
        if "image" in self.contents and self.data is not None and len(self.data) > 0:
            # 1. 处理 dtype 转换 (映射到 uint8)
            working_data = self.data
            if working_data.dtype != np.uint8:
                d_min = working_data.min()
                d_max = working_data.max()
                if d_max > d_min:
                    working_data = (
                        (working_data - d_min) / (d_max - d_min) * 255
                    ).astype(np.uint8)
                else:
                    working_data = np.zeros_like(working_data, dtype=np.uint8)
            else:
                working_data = working_data.astype(np.uint8)

            # 2. 准备色彩转换映射
            bayer_map = {
                "RGGB": cv2.COLOR_BayerRG2BGR,
                "BGGR": cv2.COLOR_BayerBG2BGR,
                "GRBG": cv2.COLOR_BayerGR2BGR,
                "GBRG": cv2.COLOR_BayerGB2BGR,
            }

            # 3. 确定视频参数
            if self.image_height is None or self.image_width is None:
                # 重新从数据获取尺寸
                h, w = working_data.shape[1], working_data.shape[2]
            else:
                h, w = self.image_height, self.image_width

            # 只有 A 模式是单通道，RGB/BGR/Bayer 最终都输出三通道 BGR
            is_color_out = self.color != "A"

            # 检查编码器并创建对象
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(mp4_path, fourcc, 25.0, (w, h), is_color_out)
            try:
                fps = 1 / np.mean(self.frame_exposure)
            except (TypeError, ValueError):
                fps = 25.0
            if not out.isOpened():
                out = cv2.VideoWriter(
                    mp4_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w, h),
                    is_color_out,
                )

            # 4. 逐帧处理并写入
            for frame in working_data:
                if self.color == "RGB":
                    frame_out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif self.color is not None and self.color in bayer_map:
                    frame_out = cv2.cvtColor(frame, bayer_map[self.color])
                else:
                    # BGR 或 A 或其他
                    frame_out = frame
                out.write(frame_out)
            out.release()

        # --- 3. 保存 PNG (Mask 和 Signal) ---
        # 逻辑：Mask (bit 6, val 64) + Signal (bit 7, val 128)
        has_mask = self.mask_frame is not None
        has_signal = self.signal_frame is not None

        if has_mask or has_signal:
            if self.image_height is None or self.image_width is None:
                # 尝试从 signal 或 mask 获取尺寸
                if has_mask:
                    h, w = self.mask_frame.shape  # type: ignore
                elif has_signal:
                    h, w = self.signal_frame.shape  # type: ignore
                else:
                    h, w = 0, 0
            else:
                h, w = self.image_height, self.image_width

            mixed_data = np.zeros((h, w), dtype=np.uint8)
            if has_mask:
                mixed_data += (self.mask_frame > 0).astype(np.uint8) * 64  # type: ignore
            if has_signal:
                mixed_data += (self.signal_frame > 0).astype(np.uint8) * 128  # type: ignore

            # 使用 png 压缩保存 (无损)
            cv2.imwrite(png_path, mixed_data, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    @classmethod
    def from_files(cls: Type[T], path_in: str) -> T:
        """
        从一组文件（JSON, MP4, PNG）读取并恢复 MeteorObservation 实例。

        Args:
            path_in: 任意一个文件路径 (json/mp4/png) 或基础文件名。

        Returns:
            MeteorObservation: 恢复的观测对象。

        Raises:
            FileNotFoundError: 如果元数据 JSON 文件不存在。
        """
        obs = cls()
        file_root = os.path.splitext(path_in)[0]
        json_path = file_root + ".json"
        mp4_path = file_root + ".mp4"
        png_path = file_root + ".png"

        # 检查必要文件
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing metadata file: {json_path}")

        # --- 1. 读取 JSON ---
        with open(json_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        # 恢复基础属性 (None 已经在初始化中设置，存在的覆盖，不存在的保持 None)
        for k, v in meta_dict.items():
            if k == "objects":
                continue  # 特殊处理
            target_key = f"_{k}" if hasattr(obs, f"_{k}") else k
            # 尝试恢复 numpy 数组 (简单的列表转数组)
            # 这里根据字段名特征来判断是否转 numpy
            if isinstance(v, list) and k in [
                "frame_time",
                "frame_exposure",
                "star_pixel_coord",
                "star_eci_coord",
                "star_magnitude",
                "star_flux",
            ]:
                setattr(obs, target_key, np.array(v))
            else:
                setattr(obs, target_key, v)

        # 恢复 Objects
        if "objects" in meta_dict:
            obs.objects = []
            for obj_dict in meta_dict["objects"]:
                met = ObservationObject()
                for k, v in obj_dict.items():
                    # 同样的数组恢复逻辑
                    if isinstance(v, list) and k in [
                        "meteor_pixel_coord",
                        "meteor_eci_coord",
                        "meteor_flux",
                        "meteor_magnitude",
                    ]:
                        setattr(met, k, np.array(v))
                    else:
                        setattr(met, k, v)
                obs.objects.append(met)

        # --- 2. 读取 MP4 ---
        if os.path.exists(mp4_path) and "image" in obs.contents:
            cap = cv2.VideoCapture(mp4_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # OpenCV 读取默认为 BGR
                # 如果是单通道模式 (mode=gray 或 color=A)，需要转灰度
                # 如果是彩色 RGB，需要 BGR2RGB
                if obs.color == "A" or obs.color is None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                elif obs.color in ["RGB", "RGGB"]:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frames.append(frame)
            cap.release()

            if frames:
                obs.data = np.array(frames)
                obs.frame_count = len(frames)
                obs.image_height = frames[0].shape[0]
                obs.image_width = frames[0].shape[1]

                # --- 新增逻辑：强制重置色彩和精度属性 ---
                # 如果读取的数组是 3 维 (F, H, W)，则是灰度；4 维 (F, H, W, 3) 则是彩色
                if len(obs.data.shape) == 3:
                    obs.color = "A"
                else:
                    obs.color = "BGR"

        # --- 3. 读取 PNG ---
        if os.path.exists(png_path):
            mixed_data = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            if mixed_data is not None:
                # 提取 Mask (bit 6)
                obs.mask_frame = (mixed_data & 64) > 0
                # 提取 Signal (bit 7)
                obs.signal_frame = (mixed_data & 128) > 0
                # 如果没有视频数据，至少确保宽高正确
                obs.image_height, obs.image_width = mixed_data.shape

        return obs

    def check_format(self) -> Union[bool, List[str]]:
        """
        检查数据完整性和一致性。

        Returns:
            bool: 如果检查通过返回 True。

        Raises:
            ValueError: 如果检查失败，包含错误详情。
        """
        errors: List[str] = []

        # 1. 基础 contents 检查
        if not self.contents:
            errors.append("Contents list is empty.")
            return errors

        # 2. 图像检查
        if "image" in self.contents:
            if self.data is None:
                errors.append("Content includes 'image' but 'data' is None.")
            else:
                d_shape = self.data.shape
                # 无论 color 是什么，H 和 W 永远在索引 1 和 2
                h, w = d_shape[1], d_shape[2]

                # 校验 Mask/Signal 时，只检查前两个维度
                if self.mask_frame is not None:
                    # 使用 .shape[:2] 确保只取 (H, W)，即便 mask 以后不小心变成了 3 维也能兼容
                    if self.mask_frame.shape[:2] != (h, w):
                        errors.append(
                            f"Mask shape {self.mask_frame.shape[:2]} != data {(h, w)}."
                        )
                if self.signal_frame is not None:
                    if self.signal_frame.shape[:2] != (h, w):
                        errors.append(
                            f"Signal frame shape {self.signal_frame.shape[:2]} mismatches image {(h, w)}."
                        )

        # 3. 时间检查
        if "time" in self.contents:
            if self.frame_time is None:
                errors.append("Content includes 'time' but 'frame_time' is None.")
            elif (
                self.frame_count is not None
                and len(self.frame_time) != self.frame_count
            ):
                errors.append(
                    f"frame_time length ({len(self.frame_time)}) != frame_count ({self.frame_count})."
                )

        # 4. 恒星检查
        if "star" in self.contents:
            if self.star_pixel_coord is None:
                errors.append("Content includes 'star' but 'star_pixel_coord' is None.")
            else:
                # 检查 2xN
                if self.star_pixel_coord.shape[0] != 2:
                    errors.append(
                        f"star_pixel_coord must be 2xN, got shape {self.star_pixel_coord.shape}."
                    )

                num_stars = self.star_pixel_coord.shape[1]

                # 检查 star_name 长度
                if self.star_name is not None and len(self.star_name) != num_stars:
                    errors.append(
                        f"star_name length ({len(self.star_name)}) mismatches star count ({num_stars})."
                    )

                # 检查 ECI 3xN
                if self.star_eci_coord is not None:
                    if self.star_eci_coord.shape != (3, num_stars):
                        errors.append(
                            f"star_eci_coord shape {self.star_eci_coord.shape} mismatches expected (3, {num_stars})."
                        )

        # 5. 流星对象检查
        if "meteor" in self.contents:
            if not self.objects:
                errors.append(
                    "Content includes 'meteor' but 'objects' list is empty or None."
                )
            else:
                for idx, obj in enumerate(self.objects):
                    if obj.meteor_index is None:
                        errors.append(f"Object {idx}: meteor_index is None.")
                        continue

                    n_frames = len(obj.meteor_index)

                    # 检查 Pixel Coord (2 * N)
                    if obj.meteor_pixel_coord is not None:
                        if obj.meteor_pixel_coord.shape != (2, n_frames):
                            errors.append(
                                f"Object {idx}: pixel_coord shape {obj.meteor_pixel_coord.shape} mismatches index len {n_frames}."
                            )

                    # 检查 ECI Coord (3 * N)
                    if obj.meteor_eci_coord is not None:
                        if obj.meteor_eci_coord.shape != (3, n_frames):
                            errors.append(
                                f"Object {idx}: eci_coord shape {obj.meteor_eci_coord.shape} mismatches index len {n_frames}."
                            )

                    # 检查 Flux (N)
                    if obj.meteor_flux is not None:
                        if len(obj.meteor_flux) != n_frames:
                            errors.append(
                                f"Object {idx}: flux length {len(obj.meteor_flux)} mismatches index len {n_frames}."
                            )

                    # 检查 Magnitude (N)
                    if obj.meteor_magnitude is not None:
                        if len(obj.meteor_magnitude) != n_frames:
                            errors.append(
                                f"Object {idx}: magnitude length {len(obj.meteor_magnitude)} mismatches index len {n_frames}."
                            )

        if errors:
            raise ValueError("Format Check Failed:\n" + "\n".join(errors))
        return True
