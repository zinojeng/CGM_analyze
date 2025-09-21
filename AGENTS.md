# AGENTS 說明

## 文件目的
- 說明 CGM 數據分析應用內部的代理（Agent）角色與合作方式。
- 幫助開發者快速了解資料流程、LLM 互動要點與自訂化注意事項。
- 提供擴充深度分析或新增代理時的參考指引。

## 系統流程
1. 使用者透過 Streamlit 介面上傳 CGM 原始檔，並設定要使用的模型與 API 金鑰。
2. `split_csv` 代理將原始檔拆成事件標記與血糖讀數兩份檔案。
3. 數據清洗與基礎指標分析後，進行胰島素、餐食與圖表產製。
4. 若提供 API 金鑰，進一步觸發多個 LLM 代理完成專業解讀與建議。
5. 彙整所有輸出，於介面中展示交互式圖表與深度建議。

## Agent / 模組職責
| Agent 名稱 | 核心職責 | 關聯模組 |
| --- | --- | --- |
| UI Orchestrator | 主掌 Streamlit 介面、流程控制、指標展示 | `main.py` |
| File Splitter | 將輸入檔拆為事件與血糖兩部分 | `split_csv.py` |
| Data Metrics & Charts | 計算 CGM 量化指標、繪製 AGP 與每日聚類圖 | `glucose_analysis.py` |
| Meal & Event Analyzer | 從事件檔抽取餐食與胰島素資料並彙整統計 | `event_analysis.py` |
| Insulin Classifier | 結合使用者輸入與時間劑量推斷胰島素類型 | `insulin_input.py`, `insulin_analysis.py` |
| Variability LLM Agent | 根據 SD/CV/MAGE 生成 AGP 與低高血糖風險說明 | `agp_variability.py` |
| GRI RAG Agent | 以 RAG 搜尋參考資料並生成 GRI 深度解析 | `gri_rag.py`, `gri_plotting.py` |
| AGP RAG Agent | 掃描 `reference_AGP/` 內的 AGP 解讀資料、建立語意索引並回傳專業報告 | （規劃中：`agp_rag.py`） |
| Deep Analysis Aggregator | 匯總所有指標，提出整體建議與胰島素互動分析 | `deep_analysis.py` |

## LLM 與 API 金鑰配置
## GPT-5 調用指引
- **正確流程**：所有 GPT-5 模型使用 Responses API，訊息以 `input_text`/`output_text` block 傳遞，並將 `max_output_tokens` 設為 ≥4096 以避免僅回傳推理內容。
- **避免錯誤參數**：不要傳入 `temperature`、`modalities`、`response_format` 或 `reasoning` 等舊版欄位；這些參數會觸發 400 `invalid_request_error`。
- **空回應排查**：若仍出現 `empty_response_text`，檢視錯誤訊息中的 payload_snippet；`reasoning_tokens` 過高通常代表輸出被 `max_output_tokens` 切斷，可再提高額度。
- **回退策略**：保持 `gpt-5-mini` → `gpt-4o-mini` 的 fallback，確保主模型失敗時仍能給出結果。

- UI 側欄提供 OpenAI 與 DeepSeek 模型選擇，需對應輸入 API 金鑰。
- 未提供金鑰時僅執行離線統計與圖表；提供金鑰後才會啟動 LLM 相關代理。
- 各代理使用的模型共用同一選擇結果，故需確保金鑰可支援所選模型。

## RAG 工作流程
1. `ReferenceDatabase` 從 `reference_database/` 目錄的 PDF 讀取文本並切片。
2. `SentenceTransformer` 建立語意向量，`Annoy` 建立近鄰索引。
3. `GRIAnalyzer` 先計算 GRI 指標，再組合成查詢向量。
4. 透過 RAG 搜尋取得最相關段落，連同原始指標送入 LLM 生成專業解釋。


## AGP 百分位統計與圖型判讀策略
1. `時間對齊與資料量檢核`：對齊 CGM 讀數至 5 分鐘時間桶，確認連續≥14 天且資料捕獲率≥70%，不足時在 UI 提示僅能產生示意性報告。
2. `百分位數彙整`：對每個時間桶計算 P50（中位數）、P25/P75（IQR）與 P5/P95（IDR），並保存對應的樣本數、餐別標籤與週期資訊。
3. `圖型特徵萃取`：以 `reference_AGP/AGP example.png` 為範本，評估 IQR/IDR 寬度與趨勢線斜率，將圖型歸類為「FNIR、治療議題、行為議題或混合議題」四象限，作為後續解讀的核心特徵。
4. `高低風險偵測`：結合百分位數與整體指標——如 P75 持續高於 180 mg/dL、P5 低於 70 mg/dL、%CV > 36%、TIR/TBR/TAR 失衡——定位問題時段並量化風險強度。
5. `RAG 解譯輸出`：使用上述特徵組成查詢向量，檢索 `reference_AGP/AGP 解讀.txt` 與其他文件的相關段落，為最終的 AGP 報告生成臨床語調的建議（含治療與行為面向）。

## AGP RAG 深度解讀流程
1. `AGPReferenceDatabase` 掃描 `reference_AGP/` 內的專家解讀（如 `AGP 解讀.txt`），切分段落並整理索引原始資料。
2. 以 `SentenceTransformer` 為段落建立向量並寫入 `Annoy`/`FAISS` 索引，保留段落來源與 AGP 指標對應。
3. 將當前 CGM 的 AGP 指標、時段摘要與檢索段落送入 LLM，輸出專業版 AGP 報告與建議。

## 執行步驟
1. 安裝依賴：`pip install -r requirements.txt`
2. 啟動服務：`streamlit run main.py`
3. 在介面上傳 CGM 檔案（支援 CSV/Excel），選擇模型並輸入 API 金鑰。
4. 點擊「執行分析」，檢視指標、圖表及文字建議。

## 常見注意事項
- 檔案需包含 `Date`, `Time`, `Sensor Glucose (mg/dL)` 等必備欄位。
- 胰島素輸入面板可自訂「早/中/晚/睡」劑量，供分類依據。
- 參考資料夾需存在且包含 `Glycemia Risk Index.pdf`，否則 RAG 模組會退回基本解析。
- 若改用 DeepSeek 模型，請確認環境已安裝 `httpx` 並提供有效金鑰。

## 後續擴充建議
- 加入更多參考文獻建立複合知識庫，或改用 FAISS 以優化搜尋。
- 為不同代理加入測試案例，確保數據格式變更時仍可正確運作。
- 擴充餐食與運動標記的語意分析，強化 LLM 跨資料整合能力。
