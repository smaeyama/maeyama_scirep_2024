#!/bin/bash

# ディレクトリのパスをカレントディレクトリに設定
directory="./"

# case_dirs の配列を作成
case_dirs=()
for filepath in "$directory"/shoot_case*; do
    if [[ -f $filepath ]]; then
        filename=$(basename "$filepath")
        case_num=$(echo "$filename" | sed -n 's/shoot_case\([0-9]\{8\}\)/\1/p')
        case_dirs+=("case$case_num")
    fi
done

# 各case_dirについて処理
for case_dir in "${case_dirs[@]}"; do
    if [[ -f "$directory/sub.q_${case_dir}.001" ]]; then
        echo "ファイルが存在します: sub.q_${case_dir}.001"
    else
        echo "ファイルが存在しません: sub.q_${case_dir}.001"
        echo "コマンド './shoot_${case_dir} 1 1' を実行するには Enter を押してください。中断するには Ctrl+C を押してください。"
        read -r -p "続行: "

        echo "コマンドを実行: ./shoot_${case_dir} 1 1"
        # 実際にコマンドを実行するには以下の行のコメントを外してください。
        ./shoot_${case_dir} 1 1
    fi
done

echo "スクリプトを終了します。"

