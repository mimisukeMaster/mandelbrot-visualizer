import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from joblib import Parallel, delayed # 並列計算用
import gc

# 精度設定
width, height = 800, 800
max_iter = 200


def main():
    global width, height, max_iter, fig, ax, img_plot, executor

    # 描画範囲
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5

    # 初回の計算
    img = generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)

    # 描画セットアップ
    cmap = create_colormap()
    cmap.set_under("black")
    
    # 画像と軸の作成
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 二次元配列から画像の描画
    img_plot = ax.imshow(img, extent=(xmin, xmax, ymin, ymax), cmap=cmap, origin="lower", vmin=1)
    
    ax.set_title("Mandelbrot Set")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    fig.colorbar(img_plot, label="Iterations")

    # マウスボタンが離されたときのイベントを設定
    fig.canvas.mpl_connect("button_release_event", on_mouse_release)

    plt.show()


def generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    """
    指定範囲でマンデルブロ集合を計算

    Returns:
        img: 計算結果の二次元配列
    """
    # 指定範囲を分割
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    # [width] * height 個の空の二次元配列を用意
    img = np.zeros((height, width))
    
    def compute_plot(i, j):
        
        # 複素数を作成
        c = complex(x[j], y[i])

        # プロットを計算
        iter_count = mandelbrot(c, max_iter)

        # 発散なら補正付き計算回数、収束なら0を格納
        return iter_count if iter_count < max_iter else 0

    # 各プロット(i, j)に対する計算タスクを追加
    tasks = []
    for i in range(height):
        for j in range(width):
            task = delayed(compute_plot)(i, j)
            tasks.append(task)

    # タスクを実行して並列計算をし結果を取得
    results = Parallel(n_jobs=8)(tasks)
    
    # 二次元配列に結果を配置
    for i in range(height):
        for j in range(width):
            img[i, j] = results[i * width + j]
    
    return img


def mandelbrot(c, max_iter):
    """
    各プロットを計算

    Returns:
        発散時: 発散の度合いを表す数値
        収束時: max_iter
    """
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            # 発散の閾値を超えたら速さに応じて平滑化し返す
            return n + 1 - np.log(np.log2(abs(z)))
        z = z*z + c

    return max_iter


def create_colormap():
    """
    カラーマップ作成
    """
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0), (1, 0, 1)]
    
    # リストから256色の線形カラーマップを生成
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    return cmap


def on_mouse_release(event, width=width, height=height, max_iter=max_iter):
    """
    拡大操作時
    """
    global xmin, xmax, ymin, ymax, executor
    
    # グラフ内でのイベントでない場合は無視
    if event.inaxes != ax:  
        return

    # 新しい表示範囲を取得
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # 再計算する
    new_img = generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
    update_plot(new_img, xmin, xmax, ymin, ymax)

def update_plot(new_img, new_xmin, new_xmax, new_ymin, new_ymax):
    """
    プロットの更新

    Args:
        future (Future): 再描画範囲の計算結果
        他: 新しい画像の範囲
    """
    global ax, img_plot, fig
    
    # 新しい画像データを設定
    img_plot.set_data(new_img)
    
    # 画像の範囲を設定
    img_plot.set_extent((new_xmin, new_xmax, new_ymin, new_ymax))
    
    # 各軸の範囲を設定
    ax.set_xlim(new_xmin, new_xmax)
    ax.set_ylim(new_ymin, new_ymax)

    # 再描画
    fig.canvas.draw_idle()

    # メモリリーク回避
    gc.collect()

if __name__ == "__main__":
    main()