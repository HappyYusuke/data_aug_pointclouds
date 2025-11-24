# data_aug_pointclouds
本リポジトリは、アノテーション済みの点群データセットをデータ拡張する機能を提供します。

<br>

# Requirements
本リポジトリは以下環境でテストしました。

<br>

# Installation
本リポジトリをクローン
```bash
git clone 
```

<br>

以下Pythonパッケージをインストールしてください。
```bash
pip install numpy tqdm shapely
```

# Usage
**1. ファイルツリー設定**
<pre>
your_data
├── label 👉 ラベルファイル格納先 (.json)
└── lidar 👉 点群ファイル格納先 (.pcd)
</pre>

> [!IMPORTANT]
> 各ファイルの拡張子は以下である必要があります。
> * 点群ファイル: `.pcd (ASCⅡ)`
> * ラベルファイル: `.json`

<br>

**2. 設定編集**
Pythonスクリプト内のパラメータを直接編集してください (詳細はスクリプト内にあります)。
```bash
vim ~/data_aut_pointclouds/
```

<br>

**3. スクリプト実行**
```bash
python3 ~
```

出力先
