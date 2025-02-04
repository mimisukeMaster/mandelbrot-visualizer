import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from threading import Thread

# 描画範囲
xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5

def mandelbrot(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n + 1 - np.log(np.log2(abs(z)))  # スムーズなグラデーションのためにスケーリング
        z = z * z + c
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

def update_image():
    global img_plot, ax, fig, xmin, xmax, ymin, ymax

    new_img = generate_mandelbrot(xmin, xmax, ymin, ymax, 800, 800, 200)

    img_plot.set_data(new_img)
    img_plot.set_extent((xmin, xmax, ymin, ymax))
    
    fig.canvas.draw_idle()

def on_range_change(event):
    global xmin, xmax, ymin, ymax, ax

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # 画像更新処理をスレッドで非同期に実行
    thread = Thread(target=update_image)
    thread.start()

def main():
    global fig, ax, img_plot

    # 計算する粒度 縦横の幅をこの値で分割
    width, height = 800, 800
    max_iter = 200

    # 初回のマンデルブロ集合を計算
    img = generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
    cmap = create_colormap()
    cmap.set_under('black')

    # 描画セットアップ
    fig, ax = plt.subplots(figsize=(10, 10))
    img_plot = ax.imshow(img, extent=(xmin, xmax, ymin, ymax), cmap=cmap, origin='lower', vmin=1)
    ax.set_title("Mandelbrot Set")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    fig.colorbar(img_plot, label='Iterations')

    # 拡大されたら再描画関数を呼ぶ
    ax.callbacks.connect('xlim_changed', on_range_change)
    ax.callbacks.connect('ylim_changed', on_range_change)

    plt.show()

if __name__ == "__main__":
    main()