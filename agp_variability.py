import numpy as np
from openai import OpenAI
import matplotlib.pyplot as plt
from glucose_analysis import calculate_metrics, create_agp, analyze_hypoglycemia, analyze_hyperglycemia
import io
import base64
import pandas as pd

def agp_variability(cgm_df, api_key):
    # 創建AGP圖
    agp_fig = create_agp(cgm_df)
    
    # 將圖像轉換為base64編碼
    img_buffer = io.BytesIO()
    agp_fig.savefig(img_buffer, format='png')
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    
    # 血糖變異性和解讀技巧的知識
    glucose_analysis_knowledge = """
    血糖變異性指標和解讀技巧：

    1. 數據充足性：
       - 感測器佩戴時間應盡可能接近100%，至少≥70%。
       - 建議收集14天數據，CGM讀數至少覆蓋70%（14天中的10天）。

    2. 血糖變異性指標：
       - 標準差（SD）：SD值越大，表示血糖波動幅度越大，變異性越高。
       - 變異係數（CV）：CV值越高，表示血糖波動相對於平均血糖值的比例越高。CV可排除平均血糖值的影響，更客觀地反映血糖波動程度。
       - AGP圖表分析：
         * 中位數線：代表"通常情況下"的血糖水平。
         * 四分位距（IQR）：第25到75百分位數範圍，IQR較寬表示血糖變異性較大。
         * 第5到95百分位數範圍：範圍越寬，表示血糖變異性越大。
       - 血糖變異幅度（MAGE）：反映血糖波動的幅度。
    
    3. 解讀技巧：血糖不穩定性 (Glycemic Instability)：
    可以理解為血糖變異性中較高風險的狀態，指的是血糖水平在短時間內發生劇烈波動的傾向， 尤其側重於難以預測且可能快速發生的血糖變化。
    雖然沒有特定的指標來量化血糖不穩定性，但可以透過分析 AGP 圖表上血糖波動的幅度、頻率和趨勢來評估。
    例如，如果 AGP 圖表顯示血糖水平頻繁地大幅度上升和下降，則表示血糖不穩定性較高，更容易發生嚴重的高血糖或低血糖事件。

    4. 解讀技巧：低血糖：
    低血糖分析
    根據您提供的資料來源和我們的對話歷史，以下列出一些關於低血糖分析的重點：
    識別低血糖模式是 AGP 報告系統性審查的首要任務。
    低血糖是血糖管理的主要限制因素。
    在審查 AGP 時，請務必將這些數據與可能影響血糖模式的患者特定因素結合起來考慮，包括年齡、糖尿病病程、體重和 BMI、腎功能、患者特定的合併症、血糖治療方案劑量和胰島素給藥時間、用餐時間、運動時間和天數、不同的工作和睡眠時間表，以及零食。
    查看每日圖表以仔細檢查低血糖模式，並查看它們是否集中在週末或特殊活動日。
    觀察 AGP 圖表和每日血糖曲線，找出低血糖發生的時間和趨勢。特別注意空腹、餐前、運動後和夜間的低血糖。
    可能導致低血糖的原因包括基礎胰島素劑量過高（清晨低血糖）、胰島素作用高峰時間錯誤計算、運動後低血糖和夜間低血糖。
    如果深藍色（25% 的值）或灰色（5% 的值）曲線接觸到 70 mg/dL 線或更低，請務必小心，或者等到低血糖得到解決後再糾正高血糖。
    
    5. 解讀技巧：高血糖：
    識別高血糖模式：查看每日圖表以仔細檢查高血糖模式，並查看它們是否集中在週末或特殊活動日。確認通常的用餐時間，並探討高血糖值通常出現在餐前還是餐後。詢問週末和工作日的起床、用餐和睡覺時間的差異。
    分析高血糖的原因：高血糖可能是由於各種因素造成的，例如胰島素劑量不足或時機不對、碳水化合物攝取過多、運動量不足、壓力或疾病。
    時間在目標範圍以上 (TAR)：TAR 是指血糖讀數高於目標範圍（>180 mg/dL，或懷孕期間 >140 mg/dL）的時間量。如果感測器血糖讀數中有 ≥25% 高於 180 mg/dL，則務必了解原因。國際共識建議，對於高風險個人，目標是讀數低於 1%。
    結合患者生活方式：檢視高血糖模式時，應考慮患者的生活方式因素，例如用餐時間、運動習慣和胰島素注射時間。例如，詢問患者在高血糖讀數出現時是否有忘記服用藥物，或是否在餐前實際注射了餐時胰島素。
    行動計畫：與患者合作制定行動計畫以解決高血糖問題，其中可能包括調整胰島素劑量、改變生活方式或兩者兼而有之。

    6. 綜合解讀技巧：
       - 優先查看低血糖模式：低血糖是血糖管理的主要限制因素。
       - 分析高血糖模式：找出高血糖值出現的時間和趨勢。
       - 評估血糖變異性：注意AGP圖表上陰影區域很寬的地方。
       - 綜合分析各項指標：結合AGP圖表、TIR、TBR、GMI等指標。
       - 結合患者生活方式：考慮運動、飲食、藥物等因素對血糖的影響。

    評估血糖變異性需綜合考慮多種指標，AGP圖表提供了直觀且全面的分析方法。
    """
    
    # 分析低血糖和高血糖
    hypo_analysis = analyze_hypoglycemia(cgm_df)
    hyper_analysis = analyze_hyperglycemia(cgm_df)
    
    # 使用OpenAI的GPT-4 Vision API分析圖像
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"你是一位專業的糖尿病醫生，擅長分析AGP圖表和評估血糖變異性。以下是關於血糖變異性指標和解讀技巧的知識，請用繁體中文回答：\n\n{glucose_analysis_knowledge}"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "請分析這張AGP圖表，評估血糖變異性。請關注以下幾點並給出專業的解釋：\n"
                                             "1. 每日血糖曲線的 variability，主要在 50%和90% percentile陰影區域的最寬處，將 AGP 圖表分為不同時間段（例如：早餐前、早餐後、午餐前、午餐後、晚餐前、晚餐後、睡前、夜間）和確切時間, 幾個最主要的分析。\n"
                                             "2. 每日血糖曲線的 instability評估,將 AGP 圖表分為不同時間段（例如：早餐前、早餐後、午餐前、午餐後、晚餐前、晚餐後、睡前、夜間）和確切時間，針對幾個最主要的分析。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}},
                ],
            }
        ],
        max_tokens=2000,
    )
    
    agp_analysis = response.choices[0].message.content
    
    # 使用GPT-4分析低血糖和高血糖數據
    hypo_hyper_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "你是一位專業的糖尿病醫生，擅長分析血糖數據。請用繁體中文回答。"
            },
            {
                "role": "user",
                "content": f"請分析以下低血糖和高血糖數據，並給出專業的評估和建議：\n\n"
                           f"低血糖數據：\n{hypo_analysis}\n\n"
                           f"高血糖數據：\n{hyper_analysis}\n\n"

                           f"1. 低血糖和高血糖發生的頻率和時間模式\n"


            }
        ],
        max_tokens=2000,
    )
    
    hypo_hyper_analysis = hypo_hyper_response.choices[0].message.content
    
    # 計算額外的指標
    metrics = calculate_metrics(cgm_df)
    sd = cgm_df['Sensor Glucose (mg/dL)'].std()
    cv = metrics['CV']
    mage = metrics['MAGE']
    
    return agp_analysis, hypo_hyper_analysis, sd, cv, mage

# 移除 calculate_mage 函數，因為它現在在 glucose_analysis.py 中