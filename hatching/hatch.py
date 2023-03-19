import math
import random
from typing import Any, Iterable, Tuple, Union

import cv2
import matplotlib.collections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import svgwrite as svgwrite
import shapely.ops
from shapely.geometry import LinearRing, MultiLineString, Polygon
from skimage import measure


def _build_circular_hatch(
    delta: float, offset: float, w: int, h: int, center: Tuple[float, float] = (0.5, 0.5)
):
    center_x = w * center[0]
    center_y = h * center[1]

    ls = []
    # If center_x or center_y > 1, ensure the full image is covered with lines
    max_radius = max(math.sqrt(w**2 + h**2), math.sqrt(center_x**2 + center_y**2))

    for r in np.arange(offset, max_radius, delta):
        # make a tiny circle as point in the center
        if r == 0:
            r = 0.001

        # compute a meaningful number of segment adapted to the circle's radius
        n = max(20, r)
        t = np.arange(0, 1, 1 / n)

        # A random phase is useful for circles that end up unmasked. If several such circles
        # start and stop at the same location, a disgraceful pattern will emerge when plotting.
        phase = random.random() * 2 * math.pi

        data = np.array(
            [
                center_x + r * np.cos(t * math.pi * 2 + phase),
                center_y + r * np.sin(t * math.pi * 2 + phase),
            ]
        ).T
        ls.append(LinearRing(data))

    mls = MultiLineString(ls)

    # Crop the circle to the final dimension
    p = Polygon([(0, 0), (w, 0), (w, h), (0, h)])
    return mls.intersection(p)


def _build_diagonal_hatch(delta: float, offset: float, w: int, h: int, angle: float = 45):
    # Keep angle between 0 and 180
    angle = angle % 180
    # Convert angle to rads
    angle_rad = angle * math.pi / 180

    lines = []
    # Draw vertical lines
    if angle == 90:
        for i in np.arange(offset, w + 1, delta):
            start = (i, 0)
            stop = (i, h)
            lines.append([start, stop])

    # Draw horizontal lines
    elif angle == 0:
        for j in np.arange(offset, h + 1, delta):
            start = (0, j)
            stop = (w, j)
            lines.append([start, stop])

    elif angle < 90:
        for i in np.arange(offset, h / math.tan(angle_rad) + w + 1, delta):
            j = abs(i * math.tan(angle_rad))

            if i <= w:
                start = (i, 0)
            else:
                start = (w, (i - w) * j / i)

            if j <= h:
                stop = (0, j)
            else:
                stop = ((j - h) * i / j, h)

            lines.append([start, stop])

    else:
        for i in np.arange(h / math.tan(angle_rad) + offset, w + 1, delta):
            j = abs((w - i) * math.tan(math.pi - angle_rad))

            if i >= 0:
                start = (i, 0)
            else:
                start = (0, -i * j / (w - i))

            if j >= h:
                stop = (w - (j - h) * (w - i) / j, h)
            else:
                stop = (w, j)

            lines.append([start, stop])
    return lines


def _plot_poly(geom, colspec=""):
    plt.plot(*geom.exterior.xy, colspec)
    for i in geom.interiors:
        plt.plot(*i.xy, colspec)


def _plot_geom(geom, colspec=""):
    if geom.geom_type == "Polygon":
        _plot_poly(geom, colspec)
    elif geom.geom_type == "MultiPolygon":
        for p in geom:
            _plot_poly(p, colspec)


def _build_mask(cnt):
    lr = [LinearRing(p[:, [1, 0]]) for p in cnt if len(p) >= 4]

    mask = None
    for r in lr:
        if mask is None:
            mask = Polygon(r)
        else:
            if r.is_ccw:
                mask = mask.union(Polygon(r).buffer(0.5))
            else:
                mask = mask.difference(Polygon(r).buffer(-0.5))

    return mask


def _save_to_svg(file_path: str, w: int, h: int, coords) -> None:
    dwg = svgwrite.Drawing(file_path, size=(w, h), profile="tiny", debug=False)

    # dwg.add(
    #     dwg.path(
    #         " ".join(
    #             " ".join(
    #                 ("M" + " L".join(f"{x},{y}" for x, y in ls.coords)) for ls in mls.geoms
    #             )
    #             for mls in vectors
    #         ),
    #         fill="none",
    #         stroke="black",
    #     )
    # )

    lines = 0
    for points in coords:
        lines += 1
        dwg.add(dwg.line(points[0], points[1], stroke="black"))
    print(lines)
    dwg.save()


def change_direction(array):
    return array[::-1]


def _arrange(coords, angle):
    angle = np.radians(angle)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    points = coords.copy().reshape(-1, 2).T
    points = R @ points
    points = points.T.reshape(-1, 2, 2)
    idxs = points[:, 0, 1]
    sorted_idxs = np.argsort(idxs)
    coords = coords[sorted_idxs]
    if True:
        reverse_flag = False
        last = idxs[0]
        for i in range(len(idxs)):
            if idxs[i] != last:
                reverse_flag = not reverse_flag
            if reverse_flag:
                coords[i] = coords[i, ::-1]
            last = idxs[i]
    return coords


def _get_coords(vectors: Iterable[MultiLineString]):
    lines = []
    for mls in vectors:
        for ls in mls.geoms:
            lines.append([ls.coords[0], ls.coords[1]])

    return np.array(lines)


def _normalize(coords, config):
    max_y = coords[:,:,1].max()
    min_y = coords[:,:,1].min()
    max_x = coords[:,:,0].max()
    min_x = coords[:,:,0].min()

    coords -= np.array([(min_x+max_x)/2, (min_y+max_y)/2]).reshape(1,1,2)

    w = float(config['page_width'] - 2 * config["margin"])
    h = float(config['page_height'] - 2 * config["margin"])
    mini = np.minimum(w/(max_x-min_x), h/(max_y-min_y))
    coords *= np.array([mini, mini]).reshape(1,1,2)

    return coords


def _save_gcode(file_path: str, config, coords):
    stack = []
    coords = _normalize(coords, config)

    offsets = np.array([config['page_width']/2 + config['x_offset'],
                        config['page_height']/2 + config['y_offset']]).reshape(1, 1, 2)
    coords += offsets

    line1_start = np.array([0, 0]).reshape(1, 1, 2)
    line2_start = np.array([config["width"], 0]).reshape(1, 1, 2)

    line1 = np.sqrt(np.sum((coords - line1_start)**2, axis=-1, keepdims=True))
    line2 = np.sqrt(np.sum((coords - line2_start)**2, axis=-1, keepdims=True))
    lines = np.concatenate((line1, line2), axis=-1)

    lines = (lines * 10).astype(int)

    gcode = open(file_path, "w")
    # gcode.write(f"C02,{round(config['pen_width'], 1)},END\n") # CMD_SETPENWIDTH 
    # gcode.write(f"C31,{round(config['motor_speed'], 1)},END\n") # CMD_SETMOTORSPEED 
    # gcode.write(f"C32,{round(config['motor_accel'], 1)},END\n") # CMD_SETMOTORACCEL 

    gcode.write(f"C09,{int(config['home_l1']*10)},{int(config['home_l2']*10)},END\n")
    gcode.write("C14,END\n")
    for points in lines:
        (l1, l2) = points[0]
        gcode.write(f"C17,{l1},{l2},{int(config['max_segment_length'])},END\n")
        gcode.write("C13,END\n")
        for (l1, l2) in points[1:]:
            gcode.write(f"C17,{l1},{l2},{int(config['max_segment_length'])},END\n")
        gcode.write("C14,END\n")
    gcode.close()


def _load_image(
    file_path: Union[str, np.ndarray],
    blur_radius: int = 10,
    image_scale: float = 1.0,
    interpolation: int = cv2.INTER_LINEAR,
    h_mirror: bool = False,
    invert: bool = False,
) -> np.ndarray:
    # Load the image, resize it and apply blur
    if isinstance(file_path, str):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = file_path
    scale_x = int(img.shape[1] * image_scale)
    scale_y = int(img.shape[0] * image_scale)
    img = cv2.resize(img, (scale_x, scale_y), interpolation=interpolation)
    if blur_radius > 0:
        img = cv2.blur(img, (blur_radius, blur_radius))

    if h_mirror:
        img = np.flip(img, axis=1)

    if invert:
        img = 255 - img

    return img

def _build_hatch(
    img: np.ndarray,
    hatch_pitch: float = 5.0,
    levels: Union[int, Tuple[int, ...]] = (64, 128, 192),
    circular: bool = False,
    center: Tuple[float, float] = (0.5, 0.5),
    invert: bool = False,
    hatch_angle: float = 45,
) -> Tuple[MultiLineString, Any, Any, Any]:

    if invert:
        levels = tuple(255 - i for i in reversed(levels))
        

    h, w = img.shape
    n_levels = len(levels)

    # border for contours to be closed shapes
    r = np.zeros(shape=(img.shape[0] + 2, img.shape[1] + 2))
    r[1:-1, 1:-1] = img

    contours = [measure.find_contours(r, levels[i]) for i in range(n_levels)]

    mls = [MultiLineString(np.empty(shape=(0, 2, 2))) for i in range(n_levels)]

    try:
        mask = [_build_mask(i) for i in contours[::-1]]

        # Spacing considers interleaved lines from different levels
        delta_factors = [2 ** (n_levels - 1)]
        delta_factors.extend([2 ** (n_levels - i) for i in range(1, n_levels)])
        offset_factors = [0]
        offset_factors.extend([2 ** (n_levels - i - 1) for i in range(1, n_levels)])

        if circular:
            lines = [
                _build_circular_hatch(
                    delta_factors[i] * hatch_pitch,
                    offset_factors[i] * hatch_pitch,
                    w,
                    h,
                    center=center,
                )
                for i in range(len(levels))
            ]
        else:
            # correct offset to ensure desired distance between hatches
            if hatch_angle % 180 != 0:
                hatch_pitch /= math.sin((hatch_angle % 180) * math.pi / 180)

            lines = [
                _build_diagonal_hatch(
                    delta_factors[i] * hatch_pitch,
                    offset_factors[i] * hatch_pitch,
                    w,
                    h,
                    angle=hatch_angle,
                )
                for i in range(n_levels)
            ]

        frame = Polygon([(3, 3), (w - 6, 3), (w - 6, h - 6), (3, h - 6)])

        mls_ = [
            MultiLineString(MultiLineString(lines[i]).difference(mask[i]).intersection(frame))
            for i in range(n_levels)
        ]

        mls = [
            MultiLineString(shapely.ops.linemerge([i for i in mls_[j].geoms]))
            for j in range(n_levels)
        ]

    except Exception as exc:
        print(f"Error: {exc}")

    all_lines = []
    for i in range(n_levels):
        all_lines.extend([ls for ls in mls[i].geoms])

    return (MultiLineString(all_lines), *contours)


def hatch(
    file_path,
    output_path: str = ".",
    board_config: dict = {},
    hatch_pitch: float = 5,
    levels: Union[int, Tuple[int, ...]] = (64, 128, 192), # 'auto'
    blur_radius: int = 10,
    image_scale: float = 1.0,
    interpolation: int = cv2.INTER_LINEAR,
    arrange: bool = True,
    h_mirror: bool = False,
    invert: bool = False,
    circular: bool = False,
    center: Tuple[float, float] = (0.5, 0.5),
    hatch_angle: float = 45,
    show_plot: bool = False,
    save_svg: bool = True,
    save_gcode: bool = True,
) -> MultiLineString:

    img = _load_image(
        file_path=file_path,
        blur_radius=blur_radius,
        image_scale=image_scale,
        interpolation=interpolation,
        h_mirror=h_mirror,
        invert=invert,
    )

    mls, *cnts = _build_hatch(
        img,
        hatch_pitch=hatch_pitch,
        levels=levels,
        invert=invert,
        circular=circular,
        center=center,
        hatch_angle=hatch_angle,
    )

    coords = _get_coords([mls])

    if save_svg:
        # save vector data to svg file
        _save_to_svg(
            f"{output_path}/output.svg", img.shape[1], img.shape[0], coords
        )

    if arrange:
        coords = _arrange(coords, hatch_angle)

    if save_gcode:
        _save_gcode(f"{output_path}/output.txt", board_config, coords)

    if show_plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap=plt.cm.gray)

        # noinspection PyShadowingNames
        def plot_cnt(contours, spec):
            for cnt in contours:
                plt.plot(cnt[:, 1], cnt[:, 0], spec, linewidth=2)

        colors = list(mcolors.BASE_COLORS.keys())
        for i, cnt in enumerate(cnts):
            plot_cnt(cnt, f"{colors[i]}-")

        plt.subplot(1, 2, 2)

        if invert:
            plt.gca().set_facecolor((0, 0, 0))
            color = (1, 1, 1)
        else:
            color = (0, 0, 0)

        plt.gca().add_collection(
            matplotlib.collections.LineCollection(
                (ls.coords for ls in mls.geoms), color=color, lw=0.3
            )
        )

        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])

        plt.show()