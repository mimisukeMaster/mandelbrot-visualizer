import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from concurrent.futures import ProcessPoolExecutor

# 描画範囲
xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5

def mandelbrot(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n + 1 - np.log(np.log2(abs(z)))  # スムーズなグラデーションのためのスケーリング
        z = z*z + c
    return max_iter

def generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    img = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            c = complex(x[j], y[i])
            iter_count = mandelbrot(c, max_iter)
            img[i, j] = iter_count if iter_count < max_iter else 0
    
    return img

def create_colormap():
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0), (1, 0, 1)]
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    return cmap

def update_plot(future, new_bounds):
    """バックグラウンドスレッドから描画を更新"""
    global ax, img_plot, fig
    new_xmin, new_xmax, new_ymin, new_ymax = new_bounds
    new_img = future.result()
    
    # 画像データを更新
    img_plot.set_data(new_img)
    img_plot.set_extent((new_xmin, new_xmax, new_ymin, new_ymax))
    ax.set_xlim(new_xmin, new_xmax)
    ax.set_ylim(new_ymin, new_ymax)

    # 再描画
    fig.canvas.draw_idle()

def on_mouse_release(event, width=800, height=800, max_iter=200):
    global xmin, xmax, ymin, ymax, executor
    
    if event.inaxes != ax:  # グラフ内でのイベントでない場合は無視
        return

    # 新しい表示範囲を取得
    new_xmin, new_xmax = ax.get_xlim()
    new_ymin, new_ymax = ax.get_ylim()

    # バックグラウンドでマンデルブロ集合を再計算
    future = executor.submit(generate_mandelbrot, new_xmin, new_xmax, new_ymin, new_ymax, width, height, max_iter)
    future.add_done_callback(lambda fut: update_plot(fut, (new_xmin, new_xmax, new_ymin, new_ymax)))

    # グローバル変数を更新
    xmin, xmax = new_xmin, new_xmax
    ymin, ymax = new_ymin, new_ymax

def main():
    global fig, ax, img_plot, executor

    # 計算する粒度 縦横の幅をこの値で分割
    width, height = 800, 800
    max_iter = 200

    # 初回のマンデルブロ集合を計算
    img = generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
    cmap = create_colormap()
    cmap.set_under('black')

    # 描画セットアップ
    fig, ax = plt.subplots(figsize=(10, 10))
    img_plot = ax.imshow(img, extent=(xmin, xmax, ymin, ymax), cmap=cmap, origin='lower', vmin=1, aspect='auto')
    ax.set_title("Mandelbrot Set")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    fig.colorbar(img_plot, label='Iterations')

    # プロセスプールを作成
    executor = ProcessPoolExecutor(max_workers=4)

    # マウスボタンが離されたときのイベントを設定
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)

    plt.show()

if __name__ == "__main__":
    main()