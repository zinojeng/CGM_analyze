#!/usr/bin/env python3
"""
測試 OpenAI o5 模型的腳本
檢查 API 金鑰是否正常運作
"""

import os
from openai import OpenAI

def test_single_model(client, model_name, pricing_info):
    """測試單一 o5 模型"""
    try:
        print(f"🔍 正在測試 {model_name}...")
        print(f"💰 定價：輸入 ${pricing_info['input']}/1M tokens, 輸出 ${pricing_info['output']}/1M tokens, 推理 ${pricing_info['reasoning']}/1M tokens")
        print("📝 請求：寫一首關於程式碼的俳句")
        print("-" * 60)
        
        # 測試模型
        result = client.responses.create(
            model=model_name,
            input="Write a haiku about code.",
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
        )
        
        print(f"✅ {model_name} 測試成功！")
        print("📄 模型回應：")
        print("-" * 40)
        print(result.output_text)
        print("-" * 40)
        
        # 如果有推理過程，也顯示出來
        if hasattr(result, 'reasoning') and result.reasoning:
            print("🧠 推理過程：")
            print(result.reasoning)
        
        return True, result.output_text
        
    except Exception as e:
        print(f"❌ {model_name} 測試失敗：")
        print(f"錯誤類型：{type(e).__name__}")
        print(f"錯誤訊息：{str(e)}")
        
        # 提供常見錯誤的解決建議
        if "authentication" in str(e).lower() or "api_key" in str(e).lower():
            print("\n💡 解決建議：")
            print("1. 檢查你的 OPENAI_API_KEY 環境變數是否設定正確")
            print("2. 確認 API 金鑰是否有效且未過期")
            print("3. 檢查帳戶是否有足夠的額度")
        elif "model" in str(e).lower():
            print("\n💡 解決建議：")
            print(f"1. 確認 {model_name} 模型是否可用")
            print("2. 檢查你的帳戶是否有權限使用此模型")
            print("3. 此模型可能尚未對所有用戶開放")
        else:
            print("\n💡 一般建議：")
            print("1. 檢查網路連線")
            print("2. 確認 OpenAI 服務狀態")
            print("3. 重新安裝 openai 套件：pip install --upgrade openai")
        
        return False, str(e)

def test_all_o5_models():
    """測試所有 o5 模型並比較結果"""
    # 模型定價資訊
    models = {
        "gpt-5": {
            "input": "1.25",
            "output": "0.125", 
            "reasoning": "10.00"
        },
        "gpt-5-mini": {
            "input": "0.25",
            "output": "0.025",
            "reasoning": "2.00"
        },
        "gpt-5-nano": {
            "input": "0.05",
            "output": "0.005",
            "reasoning": "0.40"
        }
    }
    
    try:
        # 初始化 OpenAI 客戶端
        client = OpenAI()
        
        results = {}
        successful_tests = 0
        
        print("🚀 開始測試所有 o5 模型...")
        print("=" * 80)
        
        for model_name, pricing in models.items():
            print(f"\n{'=' * 20} {model_name.upper()} {'=' * 20}")
            success, response = test_single_model(client, model_name, pricing)
            results[model_name] = {
                "success": success,
                "response": response,
                "pricing": pricing
            }
            if success:
                successful_tests += 1
            print("=" * 60)
        
        # 顯示總結
        print(f"\n🎯 測試總結")
        print("=" * 80)
        print(f"📊 成功測試：{successful_tests}/3 個模型")
        
        if successful_tests > 0:
            print("\n✅ 成功的模型：")
            for model_name, result in results.items():
                if result["success"]:
                    pricing = result["pricing"]
                    print(f"• {model_name}: 輸入${pricing['input']}, 輸出${pricing['output']}, 推理${pricing['reasoning']} (每1M tokens)")
        
        if successful_tests < 3:
            print("\n❌ 失敗的模型：")
            for model_name, result in results.items():
                if not result["success"]:
                    print(f"• {model_name}: {result['response']}")
        
        # 性價比分析
        if successful_tests > 1:
            print("\n💡 性價比分析：")
            print("• gpt-5-nano: 最便宜，適合大量簡單任務")
            print("• gpt-5-mini: 中等價位，平衡性能與成本")  
            print("• gpt-5: 最貴但可能性能最佳，適合複雜任務")
            
        return results
        
    except Exception as e:
        print("❌ 初始化錯誤：")
        print(f"錯誤類型：{type(e).__name__}")
        print(f"錯誤訊息：{str(e)}")
        return None

def check_api_key_setup():
    """檢查 API 金鑰設定"""
    print("🔑 檢查 API 金鑰設定...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        # 只顯示前幾個和後幾個字符，保護隱私
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"✅ 找到 OPENAI_API_KEY: {masked_key}")
    else:
        print("⚠️  未找到 OPENAI_API_KEY 環境變數")
        print("請設定你的 API 金鑰：")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 OpenAI o5 模型全系列測試開始")
    print("=" * 80)
    
    # 首先檢查 API 金鑰設定
    if check_api_key_setup():
        print()
        results = test_all_o5_models()
        
        if results:
            print("\n🎉 測試完成！你現在可以根據需求選擇合適的模型：")
            print("• 預算有限 → gpt-5-nano")
            print("• 平衡需求 → gpt-5-mini") 
            print("• 最佳性能 → gpt-5")
    
    print("\n" + "=" * 80)
    print("✨ 所有測試完成")
