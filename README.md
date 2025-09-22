# CCTV映像解析アプリケーション with Grounded-SAM-2

このアプリケーションは、Grounded-SAM-2を使用してCCTV（監視カメラ）映像を解析し、指定されたオブジェクトを検出・追跡・セグメンテーションするWebアプリケーションです。

## 主要機能

- **動画アップロード**: CCTV映像（MP4形式）をWebインターフェースからアップロード
- **テキストプロンプト検出**: 自然言語でオブジェクトを指定して検出
- **オブジェクト追跡**: SAM 2を使用した高精度な動画内オブジェクト追跡
- **セグメンテーション**: 検出されたオブジェクトの精密なマスク生成
- **結果可視化**: 解析結果の動画とJSON形式での保存

## 技術スタック

- **Backend**: Flask/FastAPI (Python)
- **Frontend**: HTML/CSS/JavaScript
- **AI Models**: 
  - Grounding DINO (オブジェクト検出)
  - SAM 2 (セグメンテーション・追跡)
- **Video Processing**: OpenCV, FFmpeg

## プロジェクト構造

```
webapp/
├── backend/                 # バックエンドAPI
├── frontend/               # フロントエンドUI
├── models/                 # AIモデルコード
├── grounded_sam2/          # Grounded-SAM-2統合モジュール
├── uploads/                # アップロードされた動画
├── outputs/                # 解析結果
├── static/                 # 静的ファイル
├── templates/              # HTMLテンプレート
├── checkpoints/            # AIモデルチェックポイント
├── tests/                  # テストコード
└── requirements.txt        # 依存関係
```

## セットアップと使用方法

### 1. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 2. モデルチェックポイントのダウンロード
```bash
cd checkpoints
bash download_ckpts.sh
```

### 3. アプリケーションの起動
```bash
python backend/app.py
```

### 4. Webインターフェースへアクセス
ブラウザで `http://localhost:5000` にアクセス

## 使用例

1. CCTV動画をアップロード
2. 検出したいオブジェクトをテキストで指定（例：「person」「car」「bag」）
3. 解析開始
4. 結果の確認とダウンロード

## API仕様

- `POST /upload` - 動画アップロード
- `POST /analyze` - 動画解析実行
- `GET /results/<id>` - 解析結果取得
- `GET /status/<id>` - 処理状況確認

## 注意事項

- CUDA対応GPUが推奨（CPUでも動作可能だが処理時間が長くなります）
- 大きな動画ファイルの処理には時間がかかる場合があります
- 初回起動時にモデルのダウンロードが必要です