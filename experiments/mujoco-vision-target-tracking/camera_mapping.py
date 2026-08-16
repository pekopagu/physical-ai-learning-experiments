from __future__ import annotations

"""固定正投影カメラの画素座標とMuJoCo平面座標を相互変換する。"""


IMAGE_SIZE = 64
CAMERA_CENTER_X = 0.0
CAMERA_CENTER_Y = 0.45
ORTHOGRAPHIC_HEIGHT = 2.30


def pixel_to_world(pixel_x: float, pixel_y: float) -> tuple[float, float]:
    # 画像の原点は左上だが、MuJoCo平面では上方向を+yとして扱うためy軸を反転する。
    # 0.5は画素の左上ではなく中心位置を世界座標へ対応させる補正である。
    scale = ORTHOGRAPHIC_HEIGHT / IMAGE_SIZE
    world_x = CAMERA_CENTER_X + (pixel_x + 0.5 - IMAGE_SIZE / 2) * scale
    world_y = CAMERA_CENTER_Y + (IMAGE_SIZE / 2 - pixel_y - 0.5) * scale
    return world_x, world_y


def world_to_pixel(world_x: float, world_y: float) -> tuple[float, float]:
    # 推定した世界座標を比較画像へ描画するため、pixel_to_worldの逆変換を行う。
    scale = IMAGE_SIZE / ORTHOGRAPHIC_HEIGHT
    pixel_x = (world_x - CAMERA_CENTER_X) * scale + IMAGE_SIZE / 2 - 0.5
    pixel_y = IMAGE_SIZE / 2 - (world_y - CAMERA_CENTER_Y) * scale - 0.5
    return pixel_x, pixel_y
