import numpy as np
import GPy
from scipy.spatial import distance
from scipy.stats import norm

def calculate_mahalanobis_scores(X_pool, X_data, inv_covariance_matrix=None, min_X_data_distance=None):
    """
    サンプリング候補と既存データとの多様性スコア（マハラノビス距離に基づく）を計算します。

    Parameters:
        X_pool (np.ndarray): サンプリング候補の行列
        X_data (np.ndarray): 既存データの行列
        inv_covariance_matrix (np.ndarray): X_dataに基づく共分散行列の逆行列
        min_X_data_distance (float): X_dataに基づく最小マハラノビス距離。規格化に用いる

    Returns:
        np.ndarray: サンプリング候補の多様性スコアの配列
    """
    # 共分散行列の逆行列と最小マハラノビス距離の計算（必要に応じて）
    if inv_covariance_matrix is None:
        covariance_matrix = np.cov(X_data.T)
        #inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        inv_covariance_matrix = np.linalg.pinv(covariance_matrix)
    if min_X_data_distance is None:
        X_data_distances = np.array([distance.mahalanobis(x, y, inv_covariance_matrix) 
                                     for x in X_data for y in X_data if not np.array_equal(x, y)])
        min_X_data_distance = np.min(X_data_distances) if len(X_data_distances) > 0 else 1

    # マハラノビス距離の計算と多様性スコアの算出
    distances = np.array([distance.mahalanobis(x, data_point, inv_covariance_matrix) 
                          for x in X_pool for data_point in X_data])
    distances = distances.reshape(X_pool.shape[0], len(X_data))
    min_distances = np.min(distances, axis=1)
    # diversity_scores = min_distances / min_X_data_distance
    diversity_scores = min_distances / (min_distances + min_X_data_distance)
    return diversity_scores


def calculate_strategy_scores(X_pool, model, strategy):
    """
    クエリ戦略に基づいたサンプリング候補のスコアを計算します。

    Parameters:
        X_pool (np.ndarray): サンプリング候補の特徴ベクトルの行列
        model: ガウス過程モデル
        strategy (str): クエリ戦略。"max_EI" または "uncertainty" を選択します.

    Returns:
        np.ndarray: サンプリング候補のスコアの配列
    """
    if strategy == "max_EI":
        scores = calculate_ei(X_pool, model)
    elif strategy == "uncertainty":
        _, Y_var_pool = model.predict(X_pool)
        Y_std_pool = np.sqrt(Y_var_pool)
        scores = Y_std_pool
    else:
        raise ValueError("Invalid strategy. Choose 'max_EI' or 'uncertainty'.")
    return scores


def calculate_ei(X_pool, model, xi=0.01):
    """
    Expected Improvement (EI) を計算します。

    Parameters:
        X_pool (np.ndarray): サンプリング候補の行列
        model: ガウス過程モデル
        xi (float): 探索と活用のトレードオフを制御するハイパーパラメータ

    Returns:
        np.ndarray: サンプリング候補のEIスコアの配列
    """
    mean, std = model.predict(X_pool)
    y_max = np.max(model.Y)
    z = (mean - y_max - xi) / std
    ei = (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)
    return ei


def propose_next_sampling_X(X, bounds, model, batch_size, strategy, num_candidates, Xpool=None):
    """
    クエリ戦略に基づいて新しいサンプリング位置を提案します。batch_size > 1 で複数位置提案する場合はマハラノビス距離に基づく多様性を考慮します。

    Parameters:
        X (np.ndarray): 既存データの行列
        bounds (list of tuple): 定義域のリスト
        model: ガウス過程モデル
        batch_size (int): 提案する新しいサンプリング位置の数
        strategy (str): クエリ戦略。"max_EI" または "uncertainty" を選択します.
        num_candidates (int): サンプリング候補の数

    Returns:
        np.ndarray: 新しいサンプリング位置の行列
    """
    if Xpool is None:
        # 新しいサンプリング候補の数をバッチサイズの2倍以上に設定
        num_candidates = max(num_candidates, 2 * batch_size)

        # 定義域内でランダムにサンプリング候補を生成
        #np.random.seed(0) # for debug
        X_pool = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (num_candidates, X.shape[1]))
    else:
        num_candidates = X_pool.shape[0]

    # サンプリング候補に対してクエリ戦略に基づいたスコアの評価
    strategy_scores = calculate_strategy_scores(X_pool, model, strategy)

    # 共分散行列の逆行列とX間の最小マハラノビス距離を先んじて計算
    covariance_matrix = np.cov(X.T)
    #inv_covariance_matrix = np.linalg.inv(covariance_matrix)
    inv_covariance_matrix = np.linalg.pinv(covariance_matrix)
    X_data_distances = np.array([distance.mahalanobis(x, y, inv_covariance_matrix) 
                                 for x in X for y in X if not np.array_equal(x, y)])
    min_X_data_distance = np.min(X_data_distances) if len(X_data_distances) > 0 else 1

    # 新しいサンプリング位置の選択
    X_selected = []
    for _ in range(batch_size):
        if len(X_pool) == 0:
            break
        X_data = np.vstack(([X]+ X_selected)) # XにもX_selectedにも多様性を持つようにスコア付けに渡す
        diversity_scores = calculate_mahalanobis_scores(X_pool, X_data, inv_covariance_matrix, min_X_data_distance)
        combined_scores = strategy_scores.ravel() * diversity_scores

        idx_max_score = np.argmax(combined_scores)
        X_selected.append(X_pool[idx_max_score, :])

        # 選択されたサンプルをプールから削除
        X_pool = np.delete(X_pool, idx_max_score, axis=0)
        strategy_scores = np.delete(strategy_scores, idx_max_score, axis=0)

    return np.vstack(X_selected)
    

def fit_gaussian_process_and_propose_new_samples(X, Y, bounds=None, batch_size=5, strategy="max_EI", num_candidates=100, Xpool=None):
    """
    ガウス過程回帰モデルを適合させ、新しいサンプリング位置を提案します。

    Parameters:
        X (np.ndarray): 既存データの特徴ベクトルの行列
        Y (np.ndarray): 既存データに対するターゲット値のベクトル
        bounds (list of tuple): 特徴の定義域のリスト
        batch_size (int): 提案する新しいサンプリング位置の数
        strategy (str): クエリ戦略。"max_EI" または "uncertainty" を選択します.
        num_candidates (int): サンプリング候補の数

    Returns:
        np.ndarray: 新しいサンプリング位置の行列
    """
    # 定義域の設定（与えられていない場合はXの最大値・最小値で設定）
    if bounds is None:
        bounds = [(None, None)] * X.shape[1]
        bounds = [(np.min(X[:, i]) if low is None else low, np.max(X[:, i]) if high is None else high)
              for i, (low, high) in enumerate(bounds)]

    # ガウス過程回帰モデルの構築と最適化
    kernel = GPy.kern.RBF(input_dim=X.shape[1])
    model = GPy.models.GPRegression(X, Y, kernel)
    model.optimize(messages=False)

    # 新しいサンプリング位置の提案
    X_selected = propose_next_sampling_X(X, bounds, model, batch_size, strategy, num_candidates, Xpool)
    return X_selected, model



if __name__ == "__main__":
    # デモデータ用関数(Branin関数)の定義
    def demo_func(x):
        x0 = x[...,0]
        x1 = x[...,1]
        return (-1.275 * x0**2 / np.pi**2 + 5.0 * x0 / np.pi + x1 - 6.0)**2 + (10.0 - 5.0 / (4.0 * np.pi)) * np.cos(x0) + 10.0

    # 入力パラメータ空間の次元
    input_dim = 2
    
    # 各次元の上限と下限を設定
    bounds = [(-5, 10), (0, 15)]
        
    # ランダムにデモデータを作成
    num_samples = 100
    np.random.seed(0)
    X = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (num_samples, input_dim))
    Y = demo_func(X).reshape(-1, 1)

    # 提案してもらいたいクエリ（次のサンプリング位置）の数
    batch_size = 10

    # バッチモードのクエリ提案
    print(X.shape,Y.shape)
    # X_selected, model = fit_gaussian_process_and_propose_new_samples(X, Y, bounds, batch_size, strategy="max_EI")
    X_selected, model = fit_gaussian_process_and_propose_new_samples(X, Y, bounds, batch_size, strategy="uncertainty")
    print("提案された新しいサンプリングポイント:")
    print(X_selected)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # メッシュグリッドを作成して関数を評価
    x0 = np.linspace(bounds[0][0], bounds[0][1], 50)
    x1 = np.linspace(bounds[1][0], bounds[1][1], 50)
    x0m, x1m = np.meshgrid(x0, x1)
    xm = np.stack([x0m, x1m], axis=-1)
    f_vals = demo_func(xm)
    
    # モデルによる予測
    X_new = np.hstack([x0m.reshape(-1, 1), x1m.reshape(-1, 1)])
    Y_pred, Y_var = model.predict(X_new)
    Y_pred = Y_pred.reshape(x0m.shape)
    Y_std = np.sqrt(Y_var).reshape(x0m.shape)
    ei = calculate_ei(X_new, model)
    ei = ei.reshape(x0m.shape)
    
    # 誤差比較用2次元プロット作成の関数
    vmax = f_vals.max()
    vmin = f_vals.min()
    fig = plt.figure(figsize=(10, 6))

    # ガウス過程モデルによる予測値のプロット
    ax = fig.add_subplot(231)
    quad = ax.pcolormesh(x0m, x1m, Y_pred, vmax=vmax, vmin=vmin)
    ax.scatter(X[:, 0], X[:, 1], marker="+", color='orange')
    ax.scatter(X_selected[:, 0], X_selected[:, 1], marker="*", color='red')
    ax.set_title("GP Prediction")
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")
    fig.colorbar(quad)

    # ガウス過程モデルによる予測の標準偏差のプロット
    std_max = Y_std.max()
    ax = fig.add_subplot(232)
    quad = ax.pcolormesh(x0m, x1m, Y_std, vmax=std_max, vmin=0, cmap="ocean_r")
    ax.scatter(X[:, 0], X[:, 1], marker="+", color='orange')
    ax.scatter(X_selected[:, 0], X_selected[:, 1], marker="*", color='red')
    ax.set_title("GP std")
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")
    fig.colorbar(quad)

    # EI (Expected Improvement)のプロット
    ei_max = ei.max()
    ax = fig.add_subplot(233)
    quad = ax.pcolormesh(x0m, x1m, ei, vmax=ei_max, vmin=0, cmap="cubehelix_r")
    ax.scatter(X[:, 0], X[:, 1], marker="+", color='orange')
    ax.scatter(X_selected[:, 0], X_selected[:, 1], marker="*", color='red')
    ax.set_title("EI")
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")
    fig.colorbar(quad)


    # 実際の関数値のプロット
    ax = fig.add_subplot(234)
    quad = ax.pcolormesh(x0m, x1m, f_vals, vmax=vmax, vmin=vmin)
    ax.scatter(X[:, 0], X[:, 1], marker="+", color='orange')
    ax.scatter(X_selected[:, 0], X_selected[:, 1], marker="*", color='red')
    ax.set_title("Exact")
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")
    fig.colorbar(quad)

    # 誤差のプロット
    error = Y_pred - f_vals
    emax = max([error.max(), -error.min()])
    ax = fig.add_subplot(235)
    quad = ax.pcolormesh(x0m, x1m, error, vmax=emax, vmin=-emax, cmap="bwr")
    ax.scatter(X[:, 0], X[:, 1], marker="+", color='orange')
    ax.scatter(X_selected[:, 0], X_selected[:, 1], marker="*", color='red')
    ax.set_title("Error (GP - Exact)")
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")
    fig.colorbar(quad)

    fig.tight_layout()
    plt.show()

 
 
