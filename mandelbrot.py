import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from concurrent.futures import ProcessPoolExecutor # 並列計算用
import gc

# 精度設定
width, height = 800, 800
max_iter = 200

# 各プロットを計算
def mandelbrot(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            # 発散の閾値を超えたら速さに応じて平滑化し返す
            return n + 1 - np.log(np.log2(abs(z)))
        z = z*z + c

    return max_iter

# 指定範囲でマンデルブロ集合を計算
def generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    
    # 指定範囲を分割
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    # [width] * height 個の空の二次元配列を用意
    img = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):

            # 複素数を作成
            c = complex(x[j], y[i])
            
            # プロットを計算
            iter_count = mandelbrot(c, max_iter)
            
            # 発散なら補正付き計算回数、収束なら0を格納
            img[i, j] = iter_count if iter_count < max_iter else 0
    
    return img

# カラーマップ
def create_colormap():
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0), (1, 0, 1)]
    
    # リストから256色の線形カラーマップを生成
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    return cmap

# プロット更新
def update_plot(future, new_xmin, new_xmax, new_ymin, new_ymax):
    global ax, img_plot, fig

    # 並列計算結果を取得
    new_img = future.result()
    
    # 新しい画像データを設定
    img_plot.set_data(new_img)
    
    # 画像の表示範囲
    img_plot.set_extent((new_xmin, new_xmax, new_ymin, new_ymax))
    
    #各軸の表示範囲
    ax.set_xlim(new_xmin, new_xmax)
    ax.set_ylim(new_ymin, new_ymax)

    # 再描画
    fig.canvas.draw_idle()

    # メモリリーク回避
    gc.collect()

# 拡大操作時
def on_mouse_release(event, width=width, height=height, max_iter=max_iter):
    global xmin, xmax, ymin, ymax, executor
    
    # グラフ内でのイベントでない場合は無視
    if event.inaxes != ax:  
        return

    # 新しい表示範囲を取得
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # 並列で再計算する関数を実行
    future = executor.submit(generate_mandelbrot, xmin, xmax, ymin, ymax, width, height, max_iter)
    
    # 並列の計算が全て完了した時に実行する関数オブジェクトを登録
    # futureは関数オブジェクトの引数として自動的に代入される
    future.add_done_callback(execute_callback)

def execute_callback(fut):
    return update_plot(fut, xmin, xmax, ymin, ymax)

def main():
    global width, height, max_iter, fig, ax, img_plot, executor

    # 描画範囲
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5

    # 初回の計算
    img = generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)

    # 描画セットアップ
    cmap = create_colormap()
    cmap.set_under('black')
    
    # 画像と軸の作成
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 二次元配列から画像の描画
    img_plot = ax.imshow(img, extent=(xmin, xmax, ymin, ymax), cmap=cmap, origin='lower', vmin=1)
    
    ax.set_title("Mandelbrot Set")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    fig.colorbar(img_plot, label='Iterations')

    # プロセスプールを作成し最大8つの並列処理枠を用意
    executor = ProcessPoolExecutor(max_workers=8)

    # マウスボタンが離されたときのイベントを設定
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)

    plt.show()

if __name__ == "__main__":
    main()