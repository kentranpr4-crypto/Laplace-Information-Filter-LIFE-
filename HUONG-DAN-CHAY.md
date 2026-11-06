# Hướng dẫn chạy MCHA4400 Assignment 2 trên máy khác (Ubuntu thuần và WSL2)

Tài liệu này hướng dẫn cài môi trường, build, test và chạy ba scenario (4: bay, 5: tag giống nhau, 6: phòng) trên một máy khác.
Mọi lệnh dùng script có sẵn trong `scripts/`. Chi tiết kỹ thuật nằm trong `ENVIRONMENT.md` và `engineering-reports/`.

## 0. Đọc trước: cái gì đã kiểm chứng, cái gì chưa

| Mục | Trạng thái |
|---|---|
| Cài đặt, build, test, chạy 3 scenario trên **Ubuntu 22.04.5, x86_64, GPU NVIDIA** | ĐÃ KIỂM CHỨNG (máy phát triển) |
| Ubuntu 24.04 | CHƯA KIỂM TRA |
| Ubuntu 26.04 | CHƯA KIỂM TRA |
| WSL2 (mọi bản Ubuntu, kể cả 26.04) | CHƯA KIỂM TRA |
| Máy sạch (chưa từng cài gì) | CHƯA KIỂM TRA |
| Máy ARM (Apple Silicon, ARM Windows) | KHÔNG HỖ TRỢ: lock file chỉ có `linux-64` |
| Máy không có GPU NVIDIA | CHƯA KIỂM TRA phần vẽ offscreen |

Vì vậy các mục "Ubuntu 24.04/26.04" và "WSL2" dưới đây là **chỉ dẫn hợp lý, chưa chạy thử**.
Nếu một bước lỗi, xem mục 8 và báo lại lỗi để sửa. Đừng coi dự án là "chạy được trên mọi máy" cho tới khi có người chạy thử thành công.

"WSL 26" trong yêu cầu được hiểu là WSL2 chạy Ubuntu 26.04. Nếu bạn dùng bản khác, các bước vẫn giống nhau.

## 1. Yêu cầu máy

- CPU **x86_64**. Hệ điều hành Ubuntu 22.04 trở lên (cần glibc >= 2.34). Tài khoản người dùng thường, **không chạy bằng root**.
- Ổ đĩa trống **>= 12 GiB** (cài đặt dùng 3,8 GB, bộ nhớ đệm tạm tới khoảng 6,4 GB; script từ chối chạy nếu trống dưới 12 GiB). Thêm khoảng 3 GB cho dữ liệu video và vài GB cho build.
- RAM: máy phát triển có 15 GB. Nếu máy có ít hơn 16 GB, build với ít luồng hơn (xem mục 5).
- Mạng HTTPS tới `github.com` (micromamba, nanobench) và `conda.anaconda.org` (gói conda-forge).
- Không cần `apt` cho trình biên dịch, CMake, OpenCV, Eigen, VTK, Boost. Tất cả nằm trong một thư mục riêng `~/opt/mcha4400-a2`, không chạm vào thư viện hệ thống.

### Gói hệ thống tối thiểu (cần `sudo` trên máy đích, chỉ một lần)

```bash
sudo apt update
sudo apt install -y bash curl ca-certificates coreutils gawk sed git python3 ffmpeg
```

- `git` cần vì nanobench được tải lúc configure bằng FetchContent. (CHƯA KIỂM TRA xem `git` trong prefix có đủ thay thế không, nên cài `git` hệ thống cho chắc.)
- `ffmpeg` chỉ để kiểm tra video xuất ra bằng `ffprobe`, không bắt buộc cho build.

### Thư viện vẽ offscreen (chỉ cần nếu chạy scenario có xuất video)

Dự án vẽ VTK ngoài màn hình qua EGL. Trên máy phát triển, đường này chạy với GPU NVIDIA. Để máy khác có nhiều khả năng chạy được:

```bash
sudo apt install -y libegl1 libgl1 libgl1-mesa-dri libglx-mesa0 mesa-utils xvfb
```

CHƯA KIỂM TRA trên máy nào ngoài máy phát triển. Nếu vẽ lỗi, thử mục 8.

## 2. Chuẩn bị mã nguồn và dữ liệu

Thư mục `a2/` **không phải kho git**, và `data/.gitignore` loại trừ mọi thứ trong `data/`. Nghĩa là **video không đi theo khi `git clone`**. Phải chép dữ liệu riêng.

### Trên máy gốc: đóng gói

Đóng gói mã nguồn (không kèm build, kết quả xuất, tài liệu sinh tự động):

```bash
cd /home/dinh_tran/Downloads/MCHA4400_Assignment2
tar --exclude='a2/build' --exclude='a2/build-release' --exclude='a2/build-probe' \
    --exclude='a2/out/*' --exclude='a2/docs' --exclude='a2/data' \
    -czf a2-source.tar.gz a2
```

Đóng gói dữ liệu riêng (khoảng 2,8 GB):

```bash
tar -czf a2-data.tar.gz -C a2 data
```

Thư mục `data/` cần có đủ:
- `data/outdoor/`: `calibration.MOV`, `flight.MOV`, `flight.SRT`, `config.xml`, và `sky_*.png`, `ground_*.png` (ảnh mẫu trời/đất cho visual horizon, cắt từ `flight.MOV`).
- `data/indoor/`: `calibration.MOV`, `tag_identical.MOV`, `room.MOV`, `config.xml`.
- `camera.xml` của mỗi camera. Có thể chép theo, hoặc sinh lại ở mục 6 (mất khoảng 4 phút cho camera ngoài trời và 1 phút cho camera trong nhà trên máy phát triển).

### Trên máy đích: giải nén

```bash
mkdir -p ~/mcha4400 && cd ~/mcha4400
tar -xzf /đường/dẫn/a2-source.tar.gz
tar -xzf /đường/dẫn/a2-data.tar.gz -C a2
cd a2
ls data/outdoor data/indoor     # kiểm tra đủ file
```

Kiểm tra toàn vẹn khi chép (khuyến nghị): chạy `sha256sum a2-source.tar.gz a2-data.tar.gz` ở hai máy và so kết quả.

## 3. Cài môi trường (một lần, khoảng vài chục phút tuỳ mạng)

```bash
cd ~/mcha4400/a2
scripts/setup-env.sh
```

Script này:
1. Tải micromamba 2.10.0-0 và kiểm tra SHA-256 (sai thì dừng).
2. Tạo `~/opt/mcha4400-a2` từ lock file `scripts/conda-lock-linux-64.txt` (284 gói, URL và md5 cố định).
3. Dọn bộ nhớ đệm sau khi xong.

Phiên bản được cài: GCC 14.4.0, CMake 3.31.8, Ninja 1.13.2, OpenCV 5.0.0 (bản headless), Eigen 5.0.1, VTK 9.5.2, autodiff 1.1.2, Boost 1.88.0, doctest 2.5.3, onnxruntime 1.28.0.

Hai điểm lệch so với đề gốc, đều đã được người dùng chấp nhận: autodiff 1.1.2 (bản 1.1.3 không tồn tại) và Boost 1.88.0 (VTK 9.5.2 đòi đúng bản này).

Script từ chối chạy khi: chạy bằng root, thiếu dung lượng, hoặc thư mục `~/opt/mcha4400-a2` đã có mà không do script tạo. Có thể đổi chỗ cài bằng `MCHA4400_A2_PREFIX=/đường/dẫn`.

## 4. Kiểm tra môi trường

```bash
scripts/verify-env.sh --project
```

Kết quả mong đợi trên máy phát triển: **78 PASS, 0 FAIL**. Nếu máy đích có FAIL, dừng lại và xem mục 8 trước khi build.

## 5. Build

```bash
scripts/build.sh Release -- --target a2     # tạo build-release/a2, dùng để chạy và đo thời gian
scripts/build.sh Debug   -- --target a2     # tạo build/a2 và build/tests, dùng để chạy test
```

- Máy ít RAM (dưới 16 GB), thêm giới hạn luồng, ví dụ `scripts/build.sh Release -- -j4 --target a2`. Mặc định có thể dùng nhiều luồng và tốn nhiều RAM khi biên dịch.
- Không thêm `--target a2` thì build chạy luôn bộ test; vì có **một test cố ý hỏng** (xem mục 7), `scripts/build.sh` có thể báo thất bại dù chương trình đã build xong. Dùng `--target a2` để tránh nhầm.
- Cảnh báo `_FORTIFY_SOURCE requires compiling with optimization` ở bản Debug là vô hại.
- Lần configure đầu cần mạng để tải nanobench từ GitHub.

## 6. Calibrate hai camera (nếu chưa chép `camera.xml`)

Chạy từ thư mục build, vì đường dẫn trong lệnh là tương đối:

```bash
cd build-release
./a2 ../data/outdoor/config.xml -c      # ghi data/outdoor/camera.xml (khoảng 4 phút trên máy phát triển)
./a2 ../data/indoor/config.xml  -c      # ghi data/indoor/camera.xml  (khoảng 1 phút)
```

Kết quả trên máy phát triển: sai số tái chiếu RMS 0,667 px (ngoài trời) và 1,118 px (trong nhà). Máy khác nên ra số rất gần, nhưng không nhất thiết trùng hoàn toàn.

## 7. Chạy ba scenario

```bash
cd ~/mcha4400/a2/build-release
./a2 ../data/outdoor/flight.MOV      --scenario=4 -e     # bay: flow + horizon + altimeter
./a2 ../data/indoor/tag_identical.MOV --scenario=5 -e    # tag ArUco giống nhau
./a2 ../data/indoor/room.MOV         --scenario=6 -e     # điểm landmark trong phòng
```

Lưu ý cú pháp: viết `--scenario=5` hoặc `-s=5`. Viết `-s 5` (có dấu cách) sẽ bị từ chối.

Kết quả nằm trong `a2/out/`: `<tên>_out.MOV` (video hiển thị hai khung) và `<tên>_timing.csv` (thời gian từng khung).

Kết quả tham chiếu trên máy phát triển (Release, máy rảnh):

| Scenario | Video xuất | Số khung | Thời gian | Đánh giá |
|---|---|---|---|---|
| 4 bay | 5376×1512 | 9161 | khoảng 1542 giây | chấp nhận được, hướng bay trôi |
| 5 tag giống nhau | 3840×1080 | 1074 | khoảng 117 giây | dùng được, kích thước tag 0,15 m là giả định |
| 6 phòng | 3840×1080 | 1470 | khoảng 158 giây | **không đạt chất lượng** |

Thời gian dao động khoảng ±40% giữa các lần chạy. Máy khác sẽ khác, nhất là máy không có GPU NVIDIA.

Kiểm tra video xuất:

```bash
ffprobe -v error -count_packets -select_streams v:0 \
  -show_entries stream=width,height,nb_read_packets -of csv=p=0 ../out/flight_out.MOV
# mong đợi: 5376,1512,9161
```

### Chạy test (bản Debug)

```bash
cd ~/mcha4400/a2
scripts/build.sh Debug              # có thể báo lỗi vì test hỏng đã biết, xem dưới
scripts/test.sh                     # chạy ctest trong build/
```

Kết quả mong đợi trên máy phát triển: **244 ca, 243 đạt, 1 hỏng**. Ca hỏng là `test/src/S6WalkingCamera.cpp:296` (sai số bản đồ 0,134 m so với ngưỡng 0,10 m). Đây là hạn chế đã biết của Scenario 6 và ngưỡng được giữ nguyên có chủ ý. Nếu máy đích hỏng **nhiều hơn** ca này, đó là vấn đề thật cần xem.

## 8. Xử lý sự cố

| Triệu chứng | Hướng xử lý (chưa kiểm tra trên máy khác) |
|---|---|
| `setup-env.sh` báo thiếu dung lượng | Giải phóng ổ đĩa tới trên 12 GiB, hoặc đặt `MCHA4400_A2_MIN_FREE_GB` thấp hơn và chấp nhận rủi ro đầy ổ. |
| Tải micromamba hoặc conda thất bại | Kiểm tra proxy và truy cập `github.com`, `conda.anaconda.org`. |
| `verify-env.sh` báo FAIL | Dán nguyên đoạn FAIL khi báo lỗi. Đừng build tiếp. |
| Configure lỗi khi tải nanobench | Cần `git` và mạng tới GitHub. |
| Biên dịch bị kill hoặc treo | Hết RAM. Dùng `-- -j2` hoặc `-- -j4`. |
| Vẽ offscreen lỗi (EGL, không tạo được cửa sổ) | Thử bọc lệnh bằng màn hình ảo: `xvfb-run -a ./a2 ../data/indoor/room.MOV --scenario=6 -e`. Cần đã cài `xvfb` và `libgl1-mesa-dri`. Chưa kiểm tra đường này. |
| Chạy rất chậm | Máy không có GPU thì vẽ bằng phần mềm, có thể chậm hơn nhiều lần. |
| `cv::imshow` không có | OpenCV là bản headless. Mọi hiển thị đi qua VTK, đây là thiết kế chủ ý. |
| Chế độ `--interactive=1` hoặc `2` không mở được cửa sổ | Cần `DISPLAY`. Trên WSL2 cần WSLg (Windows 11) hoặc một X server. Chưa kiểm tra kỹ, chỉ thử sơ bộ trên máy phát triển. |

## 9. Riêng cho WSL2

Tất cả chưa kiểm tra. Lưu ý quan trọng:

1. **Dùng WSL2, không dùng WSL1.** Mở PowerShell: `wsl --list --verbose`, cột VERSION phải là 2.
2. **Đặt dự án trong hệ tệp Linux** (ví dụ `~/mcha4400`), **không đặt dưới `/mnt/c/...`**. Ổ Windows qua WSL rất chậm với hàng nghìn tệp build và 2,8 GB video, và có thể lỗi quyền thực thi.
3. **Giới hạn RAM và đĩa của WSL2.** Mặc định WSL2 dùng khoảng một nửa RAM máy. Nếu build bị kill, tạo `C:\Users\<tên>\.wslconfig` với nội dung:
   ```
   [wsl2]
   memory=12GB
   swap=8GB
   ```
   rồi chạy `wsl --shutdown` trong PowerShell và mở lại.
4. **GPU:** WSL2 có thể dùng GPU qua trình điều khiển Windows (`/usr/lib/wsl/lib`). Đường EGL/NVIDIA trên WSL2 chưa được thử. Nếu vẽ offscreen lỗi, dùng `xvfb-run -a` như mục 8.
5. **Cửa sổ tương tác** (`--interactive=1/2`) cần WSLg (có sẵn trên Windows 11 với WSL cập nhật). Chạy `wsl --update` nếu thiếu.
6. Sao chép video từ Windows vào WSL: `cp /mnt/c/Users/<tên>/.../a2-data.tar.gz ~/mcha4400/` rồi giải nén trong `~/mcha4400`.
7. Ubuntu 26.04 trên WSL: gói conda-forge tự chứa trình biên dịch và thư viện riêng nên về nguyên tắc không phụ thuộc phiên bản Ubuntu, miễn glibc >= 2.34. Phần phụ thuộc hệ thống duy nhất là driver đồ hoạ (EGL/GL) và `git`. Chưa kiểm tra.

## 10. Thư mục và file quan trọng

| Đường dẫn | Nội dung |
|---|---|
| `scripts/setup-env.sh` | cài môi trường vào `~/opt/mcha4400-a2` |
| `scripts/env.sh` | `source scripts/env.sh` để kích hoạt môi trường trong shell hiện tại |
| `scripts/verify-env.sh` | kiểm tra phiên bản và khả năng tìm thấy từng thư viện |
| `scripts/build.sh` | build Debug (vào `build/`) hoặc Release (vào `build-release/`) |
| `scripts/test.sh` | chạy ctest |
| `environment.yml`, `scripts/conda-lock-linux-64.txt` | phiên bản cố định của toàn bộ gói |
| `engineering-reports/` | `plan`, `architecture`, `test-report`, `code-review`, `performance`, `final-report` và `status.json` |
| `out/` | video xuất và thời gian từng khung |
| `ENVIRONMENT.md` | tài liệu kỹ thuật đầy đủ về môi trường |

## 11. Hạn chế đã biết của dự án

- **Scenario 6 chưa đạt:** bản đồ phòng trung bình 8,4 landmark (mục tiêu >= 30), landmark sống trung vị 7 khung, không thấy cấu trúc phòng.
- **Scenario 4:** hướng bay không quan sát được nên trôi. So với GPS (chỉ dùng kiểm chứng, không bao giờ đưa vào bộ lọc), sai số vị trí 34,8 m sau khớp tương tự đầy đủ, 56,8 m khi chỉ căn hướng.
- **Scenario 5:** kích thước tag 0,15 m là giả định nên thang đo bản đồ phụ thuộc vào đó.
- `project.txt` chỉ có một dòng, nên màu và thứ tự vẽ overlay, tham chiếu altimeter và nhiều tham số khác là **giả định** (ghi trong `engineering-reports/architecture.md`).
- Không scenario nào chạy theo thời gian thực.
- Chưa chạy Doxygen. Chưa chạy lại ASan/UBSan sau các sửa đổi cuối.
