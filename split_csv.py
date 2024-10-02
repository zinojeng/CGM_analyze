import os
import io

def split_csv(uploaded_file, output_dir):
    # 获取输入文件的基本名称（不包括路径和扩展名）
    base_name = os.path.splitext(uploaded_file.name)[0]
    
    # 构造输出文件路径
    event_file = os.path.join(output_dir, f"{base_name}_event.csv")
    sensor_glucose_file = os.path.join(output_dir, f"{base_name}_sensor_glucose.csv")

    # 读取上传文件的内容
    content = uploaded_file.getvalue().decode('utf-8').splitlines()

    # 找到所有 "Index,Date,Time" 的位置
    index_positions = [i for i, line in enumerate(content) if line.startswith("Index,Date,Time")]
    
    if len(index_positions) < 2:
        print("错误：未找到足够的 'Index,Date,Time' 标记")
        return None, None

    # 处理 event.csv
    event_lines = content[index_positions[0]:index_positions[1]]
    event_headers = event_lines[0].split(',')
    event_marker_index = event_headers.index("Event Marker") if "Event Marker" in event_headers else -1

    if event_marker_index == -1:
        print("错误：未找到 'Event Marker' 列")
        return None, None

    last_valid_event_index = len(event_lines) - 1
    for i in range(len(event_lines) - 1, 0, -1):
        columns = event_lines[i].split(',')
        if len(columns) > event_marker_index and columns[event_marker_index].strip() != '':
            last_valid_event_index = i
            break

    processed_event_lines = event_lines[:last_valid_event_index + 1]

    with open(event_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(processed_event_lines))

    # 处理 sensor_glucose.csv
    sensor_glucose_lines = content[index_positions[1]:]
    glucose_headers = sensor_glucose_lines[0].split(',')
    sensor_glucose_index = glucose_headers.index("Sensor Glucose (mg/dL)") if "Sensor Glucose (mg/dL)" in glucose_headers else -1

    if sensor_glucose_index == -1:
        print("错误：未找到 'Sensor Glucose (mg/dL)' 列")
        return None, None

    processed_glucose_lines = [sensor_glucose_lines[0]]  # 保留标题行
    for line in sensor_glucose_lines[1:]:  # 从第二行开始处理
        columns = line.split(',')
        if len(columns) > sensor_glucose_index and columns[sensor_glucose_index].strip() != '':
            processed_glucose_lines.append(line)
        else:
            break  # 停止处理，因为我们已经到达了最后一个有效的 Sensor Glucose 数据

    with open(sensor_glucose_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(processed_glucose_lines))

    print(f"数据已拆分成两个文件：{event_file} 和 {sensor_glucose_file}")
    return event_file, sensor_glucose_file

# 使用示例（不需要在实际的split_csv.py文件中包含这个）
# split_csv('path/to/input/file.csv', 'path/to/output/directory')