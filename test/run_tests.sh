#!/bin/bash

# OpenAI o5 模型測試腳本
# 使用方法：./run_tests.sh 或者 bash run_tests.sh

echo "🔧 設定測試環境..."

# 檢查是否有 API 金鑰
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  請設定你的 OpenAI API 金鑰："
    echo "export OPENAI_API_KEY='your-actual-api-key-here'"
    echo ""
    echo "然後重新執行此腳本"
    echo ""
    echo "或者直接在這裡設定 (不推薦，因為會暴露在命令歷史中)："
    read -p "輸入你的 API 金鑰 (或按 Enter 跳過): " api_key
    
    if [ ! -z "$api_key" ]; then
        export OPENAI_API_KEY="$api_key"
        echo "✅ API 金鑰已設定"
    else
        echo "❌ 未設定 API 金鑰，無法繼續測試"
        exit 1
    fi
fi

echo "🚀 啟動虛擬環境並執行測試..."
echo "================================================================================"

# 啟動虛擬環境並執行測試
source venv/bin/activate
python test_o5_model.py

echo "================================================================================"
echo "✨ 測試腳本執行完成"
