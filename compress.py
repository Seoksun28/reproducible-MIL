import os
import argparse
import tarfile

BATCH_SIZE = 6  # 한 압축파일에 들어갈 SVS 개수

def main():
    parser = argparse.ArgumentParser(description="SVS 파일을 일정 개수로 묶어 tar.gz로 압축")
    parser.add_argument("--dir", type=str, default=".", help="SVS 파일이 있는 폴더")
    parser.add_argument("--output-dir", type=str, required=True, help="압축 파일 저장 폴더")
    parser.add_argument("--prefix", type=str, default="batch", help="압축 파일 이름 prefix")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.dir)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # .svs 파일 목록 가져오기
    files = [
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.lower().endswith(".svs") and os.path.isfile(os.path.join(root_dir, f))
    ]
    files.sort()

    if not files:
        print("[WARN] .svs 파일이 없습니다.")
        return

    print(f"[INFO] SVS 파일 {len(files)}개 발견")

    # 6개씩 묶어서 압축
    batch_idx = 0
    for i in range(0, len(files), BATCH_SIZE):
        batch_idx += 1
        batch_files = files[i:i + BATCH_SIZE]
        archive_name = f"{args.prefix}_{batch_idx:03d}.tar.gz"
        archive_path = os.path.join(out_dir, archive_name)

        print(f"[+] {archive_name} 생성 (파일 {len(batch_files)}개)")
        with tarfile.open(archive_path, "w:gz") as tar:
            for path in batch_files:
                arcname = os.path.basename(path)
                print(f"    - {arcname}")
                tar.add(path, arcname=arcname)

    print("[ALL DONE] 모든 압축 완료")

if __name__ == "__main__":
    main()
