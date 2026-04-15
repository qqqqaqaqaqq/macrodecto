import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR, "app")

from app.services.processed import Processed
from app.core.config import macro_config
from app.services.train import EncoderTrain
from app.services.inference import EncoderInference

MENU = {
    # [단계 1] 사전 학습 (Self-Supervised)
    "1": ("💎  Pre-train Encoder",    EncoderTrain),      # MAE: 차트의 문법 학습
    "2": ("🔍  Verify Encoder",      EncoderInference),  # 복원 성능 확인 (아까 본 그 차트)
    "3": ("📊  View Encoder Logs",  None),              # 로그 확인

    # [유틸리티]
    "4": ("🛠️  Data Processing",     Processed),         # 데이터 전처리
    "Q": ("🚪  Exit System",         None),              # 시스템 종료
}

if __name__ == "__main__":
    print("\n" + "=" * 45)
    print(f"{'🤖  MACRO DETECTOR AI PIPELINE CONTROL':^45}")
    print("=" * 45)
    for key, (label, _) in MENU.items():
        print(f"  [{key}]  {label}")
    print("=" * 45)

    choice = input("\n▶  번호를 입력하세요: ").strip().upper()

    if choice not in MENU:
        print("❌  올바른 번호를 선택해주세요.")
    else:
        # encoder
        if choice == "1":
            data_path = os.path.join(BASE_DIR, "data", "raw", "raw.json")
            processed = Processed(config=macro_config, base_dir=BASE_DIR)
            numpy_array= processed.generate_indicators(path=data_path)
            train_loader, val_loader =processed.generation_procceed_data(numpy_array)
            
            train = EncoderTrain(
                config=macro_config, 
                base_dir=BASE_DIR, 
                input_size=processed.input_size
            )
            train.run(train_loader, val_loader)

        elif choice == "2":
            data_list = ["normal.json", "user.json"]
            error = []

            for data_type in data_list:
                data_path = os.path.join(BASE_DIR, "data", "test", f"{data_type}")
                processed = Processed(config=macro_config, base_dir=BASE_DIR)
                numpy_array= processed.generate_indicators(path=data_path)
                loader, _ =processed.generation_procceed_data(numpy_array, inference_mode=True)

                infer = EncoderInference( # 클래스명 확인 필요
                    config=macro_config, 
                    base_dir=BASE_DIR, 
                    input_size=processed.input_size
                )
                error.append(infer.run(loader))

            normal_loss = error[0]
            user_loss = error[1]

            print(f"\n" + "="*30)
            print(f"Normal Baseline Loss: {normal_loss:.6f}")
            print(f"Target User Loss:    {user_loss:.6f}")
            print("="*30)

            diff_ratio = abs(normal_loss - user_loss) / (normal_loss + 1e-9)

            threshold_ratio = 0.2 

            if diff_ratio > threshold_ratio:
                status = "ANOMALY (Anomaly Detected!)"
                if user_loss < normal_loss:
                    reason = "패턴이 너무 일정함 (매크로 의심)"
                else:
                    reason = "움직임이 비정상적으로 불규칙함"
            else:
                status = "NORMAL (Human-like)"
                reason = "정상 범주 내의 움직임"

            print(f"판정 결과: {status}")
            print(f"상세 이유: {reason} (차이: {diff_ratio*100:.2f}%)")

        elif choice == "3":
            from app.utilites.log_view import log_view
            log_view(base_dir=os.path.join(BASE_DIR, "weights", "encoder", "logs"))
            
        elif choice == "4":
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()  # 메인 창 숨기기
            
            data_path = filedialog.askopenfilename(
                title="데이터 파일 선택",
                initialdir=os.path.join(BASE_DIR, "data"),
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            print(f"✅ 선택된 파일: {os.path.basename(data_path)}")
            
            processed = Processed(
                config=macro_config,
                base_dir=BASE_DIR,
                processed_check=True
            )
            processed.generate_indicators(path=data_path)

        # --- [Q] Exit ---
        elif choice == "Q":
            print("👋  시스템을 종료합니다. 성투하세요!")
            exit()