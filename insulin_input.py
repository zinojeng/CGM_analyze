import streamlit as st
from streamlit.components.v1 import html

def get_insulin_info():
    st.write("請選擇您使用的胰島素類型（可多選）：")
    
    insulin_types = {
        "長效胰島素": ["Lantus", "Levemir", "Toujeo", "Tresiba"],
        "速效胰島素": ["Humalog", "NovoRapid", "Apidra", "Lispro"],
        "預混胰島素": ["Novomix 30", "Humalog Mix 25", "Humalog Mix 50", "Ryzodeg 30"]
    }

    insulin_info = {}
    selected_insulins = []

    # 創建兩列：一列用於胰島素選擇，一列用於劑量輸入
    col1, col2 = st.columns(2)

    with col1:
        for insulin_category, options in insulin_types.items():
            st.write(f"{insulin_category}：")
            for option in options:
                if st.checkbox(option, key=f"{insulin_category}_{option}"):
                    selected_insulins.append(option)

    with col2:
        if selected_insulins:
            st.write("劑量 (單位)：")
            
            # 添加"早中晚睡"標籤
            html("""
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <div style="width: 60px; text-align: center;">早</div>
                <div style="width: 60px; text-align: center;">中</div>
                <div style="width: 60px; text-align: center;">晚</div>
                <div style="width: 60px; text-align: center;">睡</div>
            </div>
            """, height=30)

            for option in selected_insulins:
                st.write(f"{option}:")
                html_inputs = f"""
                <div style="display: flex; justify-content: space-between;">
                    <input type="number" id="{option}_morning" value="0" style="width: 60px; text-align: center;">
                    <input type="number" id="{option}_noon" value="0" style="width: 60px; text-align: center;">
                    <input type="number" id="{option}_evening" value="0" style="width: 60px; text-align: center;">
                    <input type="number" id="{option}_bedtime" value="0" style="width: 60px; text-align: center;">
                </div>
                <script>
                function updateValue(id) {{
                    var input = document.getElementById(id);
                    var value = input.value;
                    var key = id + '_value';
                    var event = new CustomEvent('streamlit:setValue', {{
                        detail: {{ key: key, value: value }}
                    }});
                    window.dispatchEvent(event);
                }}
                document.getElementById('{option}_morning').addEventListener('change', function() {{ updateValue('{option}_morning'); }});
                document.getElementById('{option}_noon').addEventListener('change', function() {{ updateValue('{option}_noon'); }});
                document.getElementById('{option}_evening').addEventListener('change', function() {{ updateValue('{option}_evening'); }});
                document.getElementById('{option}_bedtime').addEventListener('change', function() {{ updateValue('{option}_bedtime'); }});
                </script>
                """
                html(html_inputs, height=50)

                morning = st.session_state.get(f"{option}_morning_value", "0")
                noon = st.session_state.get(f"{option}_noon_value", "0")
                evening = st.session_state.get(f"{option}_evening_value", "0")
                bedtime = st.session_state.get(f"{option}_bedtime_value", "0")
                
                insulin_info[option] = {
                    "morning": float(morning) if morning else 0,
                    "noon": float(noon) if noon else 0,
                    "evening": float(evening) if evening else 0,
                    "bedtime": float(bedtime) if bedtime else 0
                }

    for insulin_category, options in insulin_types.items():
        insulin_info[insulin_category] = [option for option in options if option in selected_insulins]

    return insulin_info