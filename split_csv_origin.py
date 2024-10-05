def process_part1(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 找到第一个包含 "Index,Date,Time" 的行
    index_line_number = next(i for i, line in enumerate(lines) if 'Index' in line and 'Date' in line and 'Time' in line)

    # 从该行开始写入新文件
    with open(output_file, 'w', encoding='utf-8') as file:
        file.writelines(lines[index_line_number:])

    print(f"已处理 {input_file} 并保存为 {output_file}")

def split_csv(input_file, output_file1, output_file2):
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # 找到所有 "Index,Date,Time" 的位置
    index_positions = [i for i in range(len(content)) if content.startswith("Index,Date,Time", i)]
    
    if len(index_positions) < 2:
        print("错误：未找到足够的 'Index,Date,Time' 标记")
        return

    # 分割内容
    part1 = content[:index_positions[1]]
    part2 = content[index_positions[1]:]

    # 写入 part1.csv（未处理版本）
    with open(output_file1, 'w', encoding='utf-8') as file:
        file.write(part1.strip())

    # 处理 part2
    lines = part2.split('\n')
    headers = lines[0].split(',')
    sensor_glucose_index = headers.index("Sensor Glucose (mg/dL)") if "Sensor Glucose (mg/dL)" in headers else -1

    if sensor_glucose_index == -1:
        print("错误：未找到 'Sensor Glucose (mg/dL)' 列")
        return

    processed_lines = []
    for line in lines:
        if line.strip() == '':
            continue
        columns = line.split(',')
        if len(columns) > sensor_glucose_index and columns[sensor_glucose_index].strip() != '':
            processed_lines.append(line)
        else:
            break  # 停止处理，因为我们已经到达了最后一个有效的 Sensor Glucose 数据

    # 写入 part2.csv
    with open(output_file2, 'w', encoding='utf-8') as file:
        file.write('\n'.join(processed_lines))

    print(f"数据已拆分成两个文件：{output_file1} 和 {output_file2}")

# 使用函数
split_csv('140692Ho.csv', 'part1_unprocessed.csv', 'part2.csv')
process_part1('part1_unprocessed.csv', 'part1.csv')