# data_aug_pointclouds
<img src=fig/1.png width=1000>

# Description
本リポジトリは、アノテーション済みの点群データセットをデータ拡張する機能を提供します。

<br>

# Requirements
本リポジトリは以下環境でテストしました。

| 項目 | 要件 |
| --- | --- |
| Python | 3.10.12 |
| NumPy | 1.26.4 |
| tqdm | 4.67.1 |
| Shapely | 2.1.2 |

<br>

# Installation
本リポジトリをクローン
```bash
git clone https://github.com/HappyYusuke/data_aug_pointclouds.git
```

<br>

以下Pythonパッケージをインストールしてください。
```bash
pip install numpy tqdm shapely
```

<br>

# Usage
**1. ファイルツリー設定**
<pre>
your_data
├── label 👉 ラベルファイル格納先 (.json)
└── lidar 👉 点群ファイル格納先 (.pcd)
</pre>

> [!IMPORTANT]
> 各ファイルの拡張子は以下である必要があります。
> * 点群ファイル: `.pcd (ASCⅡ)` or `.bin`
> * ラベルファイル: `.json` or `.txt`

<br>

**2. 設定編集**
Pythonスクリプト内のパラメータを直接編集してください (詳細はスクリプト内にあります)。
```bash
vim ~/data_aut_pointclouds/augmentator.py
```

<br>

**3. スクリプト実行**
Pythonスクリプトを実行してください。
```bash
python3 ~/data_aut_pointclouds/augmentator.py
```

<br>

実行結果は`data_aug_pointclouds`下に出力され、以下のようになります。
<pre>
dataset_augmented
├── lidar 👉 点群ファイル (.bin)
├── label 👉 カメラ座標のラベルファイル (.txt)
└── label_lidar 👉 LiDAR座標のラベルファイル (.txt) (基本的にこちらを使用します。)
</pre>

> [!TIP]
> 結果を確認したい場合は[pointcloud_annotations](https://github.com/HappyYusuke/pointcloud_annotation.git)の[tools](https://github.com/HappyYusuke/pointcloud_annotation/tree/main/tools)にある[visualize_annotation.py](https://github.com/HappyYusuke/pointcloud_annotation/tree/main/tools#Check-the-annotations)を使用できます。
