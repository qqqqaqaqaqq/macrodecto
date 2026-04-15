import glob
import json
import os
from tqdm import tqdm

# 경로 설정 (백슬래시 이슈 방지를 위해 r-string 유지)
src_path = r"C:\Users\HONG\Desktop\tranformer\mouse\sample\pattern_game\backend\data\captured_movements"
target_path = r"C:\Users\HONG\Desktop\연구\MOUSE\app\data\raw"

def merge_json():
    # 1. 대상 경로에 폴더가 없으면 생성
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"폴더 생성됨: {target_path}")

    # 2. glob를 사용하여 모든 json 파일 목록 가져오기
    file_list = glob.glob(os.path.join(src_path, "*.json"))
    
    if not file_list:
        print("병합할 JSON 파일이 없습니다.")
        return

    merged_data = []

    # 3. tqdm으로 진행률을 확인하며 파일 읽기
    print(f"{len(file_list)}개의 파일을 병합 중입니다...")
    for file_path in tqdm(file_list, desc="Merging"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 데이터 타입에 따라 처리 (리스트면 extend, 사전이면 append)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    merged_data.append(data)
        except Exception as e:
            print(f"파일 로드 실패 ({os.path.basename(file_path)}): {e}")

    output_file = os.path.join(target_path, "merged_data.json")
    
    # 4. 기존 파일이 있으면 로드, 없으면 빈 리스트로 시작
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                origin = json.load(f)
                if not isinstance(origin, list):
                    origin = [origin] # 리스트가 아니면 리스트로 감싸줌
            except json.JSONDecodeError:
                origin = []
    else:
        origin = []

    origin.extend(merged_data)        
    output_file = os.path.join(target_path, "raw.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(origin, f, ensure_ascii=False, indent=4)

    print(f"--- 병합 완료! ---")
    print(f"저장 위치: {output_file}")
    print(f"총 데이터 수: {len(merged_data)}")

if __name__ == "__main__":
    merge_json()