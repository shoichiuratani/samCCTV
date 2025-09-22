# CCTV映像解析アプリケーション - デプロイメントガイド

## 🚀 アプリケーション概要

このアプリケーションは、Grounded-SAM-2を使用してCCTV映像を解析し、指定されたオブジェクトを検出・追跡・セグメンテーションするWebアプリケーションです。

### ✨ 主要機能

- **動画アップロード**: Web UI経由でのCCTV動画アップロード (最大512MB)
- **テキストプロンプト検出**: 自然言語による柔軟なオブジェクト指定
- **リアルタイム追跡**: SAM 2による高精度な動画内オブジェクト追跡
- **結果可視化**: 検出結果の動画とJSON形式での保存・ダウンロード
- **システム監視**: リアルタイムでの処理状況監視

## 🌐 アクセス情報

**アプリケーションURL**: https://5000-iof9jqdar4gcr4g56krsp-6532622b.e2b.dev

### 主要エンドポイント

- `/` - メインUI
- `/upload` - 動画アップロード API
- `/analyze` - 動画解析開始 API
- `/system/status` - システムステータス確認
- `/tasks` - 処理履歴一覧

## 📋 使用方法

### 1. 動画アップロード
1. メインページにアクセス
2. 動画ファイルをドラッグ&ドロップまたは選択
3. アップロード完了を待機

### 2. オブジェクト指定・解析開始
1. テキストプロンプトに検出したいオブジェクトを入力
   - 例: `person, car, bicycle`
   - 例: `人, 車, バッグ`
2. 「解析開始」ボタンをクリック
3. 処理完了まで待機

### 3. 結果確認・ダウンロード
1. 解析完了後、結果サマリーを確認
2. 「解析済み動画」または「注釈データ」をダウンロード
3. 検出結果詳細テーブルで各フレームの検出状況を確認

## 🔧 技術仕様

### アーキテクチャ
```
Frontend (HTML/CSS/JS) 
    ↓ HTTP API
Backend (Flask) 
    ↓ Python
Grounded-SAM-2 Integration
    ↓ AI Models
Video Analysis Pipeline
```

### 使用技術
- **Backend**: Flask, Python 3.12
- **Frontend**: Vanilla JavaScript, CSS3
- **AI Models**: Grounded-SAM-2 (統合済み)
- **Video Processing**: OpenCV, FFmpeg
- **Process Management**: Supervisor

### システム要件
- CPU: 最低2コア (推奨: 4コア以上)
- RAM: 最低4GB (推奨: 8GB以上)
- GPU: CUDA対応GPU推奨 (CPUでも動作可能)
- Storage: 最低10GB (動画ファイル用)

## 🛠 開発・カスタマイズ

### プロジェクト構造
```
webapp/
├── backend/           # Flask API
├── frontend/          # Web UI (静的ファイル)
├── grounded_sam2/     # AI統合モジュール
├── uploads/           # アップロード動画
├── outputs/           # 解析結果
└── logs/              # アプリケーションログ
```

### 設定ファイル
- `.env` - 環境変数設定
- `supervisord.conf` - プロセス管理設定
- `requirements.txt` - Python依存関係

### ログ確認
```bash
# アプリケーションログ
tail -f logs/app.log

# エラーログ
tail -f logs/app_error.log

# Supervisorログ
tail -f supervisord.log
```

## 🔄 実際のGrounded-SAM-2統合

現在の実装はモック（テスト用）です。本格的な運用には以下が必要：

### 1. 公式Grounded-SAM-2のクローン
```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2
```

### 2. モデルチェックポイントダウンロード
```bash
# SAM 2チェックポイント
cd checkpoints
bash download_ckpts.sh

# Grounding DINOチェックポイント  
cd ../gdino_checkpoints
bash download_ckpts.sh
```

### 3. 実装の置換
- `grounded_sam2/analyzer.py`の`MockSAM2Model`と`MockGroundingDINOModel`を実際のモデルに置換
- `detect_objects()`と`segment_object()`メソッドを実際のGrounded-SAM-2 APIに統合

### 4. GPU最適化
- CUDA環境のセットアップ
- バッチ処理の最適化
- メモリ管理の改善

## 🚨 注意事項・制限事項

### 現在の制限
- **モック実装**: 実際のAI推論は行われません（デモ用データを返します）
- **ファイルサイズ**: 最大512MB
- **処理時間**: 大きな動画は長時間かかる場合があります
- **同時実行**: 複数の解析を並行実行する際の制限

### セキュリティ
- アップロードされたファイルは適切に検証されます
- ファイルは安全なディレクトリに保存されます
- 不要なファイルは定期的にクリーンアップしてください

## 📞 サポート・問題対応

### よくある問題

1. **アップロードに失敗する**
   - ファイルサイズ (512MB以下) を確認
   - 対応形式 (MP4, AVI, MOV, MKV) を確認

2. **処理が完了しない**
   - ログを確認: `tail -f logs/app_error.log`
   - プロセス状況確認: `supervisorctl -c supervisord.conf status`

3. **結果がダウンロードできない**
   - 解析が完了しているか確認
   - ディスク容量を確認

### トラブルシューティング
```bash
# アプリケーション再起動
supervisorctl -c supervisord.conf restart cctv-app

# システム状況確認
curl http://localhost:5000/system/status

# 処理状況確認
curl http://localhost:5000/tasks
```

## 🔮 今後の拡張予定

- **リアルタイム処理**: ライブ映像ストリームの対応
- **データベース統合**: 結果の永続化
- **ユーザー管理**: 複数ユーザー対応
- **クラウド統合**: AWS/GCP/Azure対応
- **API拡張**: RESTful APIの充実

---

**開発者**: CCTV Analysis Team  
**最終更新**: 2024年9月22日  
**バージョン**: 1.0.0