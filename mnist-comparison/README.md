# MNIST: Softmax thủ công và CNN

## Chạy trên Windows / VS Code

Mở terminal tại thư mục chứa các file Python. Không cần tạo virtual environment.

```powershell
python -m pip install -r requirements.txt
python main.py --model softmax
python main.py --model cnn
python visualize.py --softmax results/softmax --cnn results/cnn
```

Nếu môi trường hiện tại đã chạy được Torch/Torchvision, có thể chỉ cần cài thêm:

```powershell
python -m pip install numpy matplotlib scikit-learn
```

Mặc định mỗi model chạy 10 epoch trên CPU; thời gian chạy phụ thuộc máy.
Ví dụ đổi số epoch: `python main.py --model softmax --epochs 15`.
Mặc định `python main.py` chạy Softmax. Chạy lại cùng model sẽ ghi đè kết quả của model đó;
để lưu một thí nghiệm riêng, dùng `--output-dir results_experiment_2`.

## Những file đã sửa và bổ sung

- `main.py`: thay file đã gửi. Chia dữ liệu, huấn luyện, chọn checkpoint, đánh giá test,
  lưu số liệu và gọi vẽ biểu đồ. Không đánh giá test mỗi epoch nữa.
- `train.py`: thay file đã gửi. Giữ Softmax với gradient thủ công, không dùng `.backward()`.
  Bỏ phần `main()` trùng lặp. Thêm hàm train riêng cho CNN bằng autograd.
  Loss dùng log-sum-exp thay vì chặn xác suất rồi log để ổn định số học.
- `visualize.py`: vẽ kết quả từng model và so sánh hai model từ các file đã lưu.
- `cnn_model.py`: CNN mẫu 2 convolution + 2 pooling + fully connected, 52.138 tham số.
  Đây là model mẫu bổ sung vì bạn chưa gửi file CNN hiện tại.
- `softmax_classifier.py`: bản tham chiếu bổ sung, 7.850 tham số. Vì bạn chỉ gửi main/train,
  chưa thể kiểm tra file classifier hiện tại. Nếu giữ file cũ, nó phải xuất lớp
  `SoftmaxClassifier(input_size, num_classes)`, có `W` shape `[784,10]`, `b` shape `[10]`,
  và `model(images)` phải trả **logits**, không trả xác suất đã softmax.
  Sao lưu file classifier cũ trước khi thay bằng bản tham chiếu.

## Quy trình và cách đọc kết quả

1. Bộ 60.000 ảnh gốc được chia ngẫu nhiên thành 50.000 train và 10.000 validation.
   Seed 42 cố định các chỉ số chia; đây là random split, không phải stratified split.
   Hai model dùng cùng split và pixel `[0,1]` qua `ToTensor()` như code gốc.
2. Mỗi epoch: cập nhật tham số chỉ bằng train, đánh giá validation không cập nhật.
   Lưu `best_model.pth` khi **validation loss thấp nhất** (nếu bằng nhau giữ epoch trước).
3. Nạp checkpoint tốt nhất rồi đánh giá 10.000 ảnh test chính thức một lượt.
   Không chọn hyperparameter theo kết quả test.
4. Tự xuất báo cáo vào `results/softmax/` hoặc `results/cnn/`.

Trong mỗi thư mục kết quả:

| File | Nội dung |
|---|---|
| `best_model.pth` | Tham số checkpoint tốt nhất và thông tin split; không phải checkpoint để resume đầy đủ optimizer |
| `history.json`, `history.csv` | Loss/accuracy train và valid từng epoch |
| `learning_curves.png` | Đường loss và accuracy; train nét liền, valid nét đứt |
| `config.json` | Seed, split hash, preprocessing, optimizer, lr, thời gian và thông số model |
| `split_indices.npz` | Chỉ số train/valid trong tập MNIST gốc |
| `predictions.npz`, `predictions.csv` | ID ảnh test, nhãn thật, nhãn dự đoán, softmax score; NPZ có cả ảnh |
| `confusion_matrix.png`, `confusion_matrix.csv` | Hàng là nhãn thật, cột là dự đoán; ô ghi số ảnh, màu là tỷ lệ trong hàng |
| `per_class_metrics.csv`, `per_class_metrics.png` | Precision, recall, F1 từng số 0–9 |
| `classification_report.json` | Báo cáo đầy đủ, gồm macro/weighted average |
| `correct_examples.png`, `wrong_examples.png` | Tối đa 12 ảnh đầu tiên thỏa điều kiện theo thứ tự test |
| `summary.csv` | Test loss, accuracy, macro F1, số tham số, thời gian |

Accuracy trong JSON/CSV là tỷ lệ từ 0 đến 1. Ví dụ 0.92 là 92%.
Macro F1 là trung bình F1 của 10 chữ số. Score softmax hiển thị cạnh ảnh không bảo đảm
là xác suất đúng đã được hiệu chỉnh.

Sau khi cả hai model chạy xong, lệnh so sánh tạo `results/comparison/` gồm:
biểu đồ chung loss/accuracy, hai confusion matrix cùng thang màu, F1 từng chữ số,
bảng tổng hợp CSV/PNG, ảnh Softmax sai/CNN đúng, Softmax đúng/CNN sai, cả hai sai.
Script kiểm tra cùng split, preprocessing, ảnh và thứ tự nhãn test trước khi so sánh.
Nếu chưa train CNN thì vẫn có toàn bộ báo cáo riêng cho Softmax.

Vẽ lại Softmax mà không huấn luyện lại:

```powershell
python visualize.py --run results/softmax
```

## Lưu ý khi trình bày

- Train accuracy/loss được cộng dồn trong quá trình trọng số thay đổi; CNN còn bật dropout.
  Validation được đo ở cuối epoch và tắt dropout. Vì vậy valid đôi khi tốt hơn train.
- Softmax dùng SGD thủ công lr=0.1; CNN dùng Adam lr=0.001. Báo cáo rõ các cấu hình này:
  so sánh phản ánh cả mô hình lẫn quy trình tối ưu, không chỉ riêng kiến trúc.
- Thời gian train cộng các lượt train epoch, gồm nạp batch, forward/backprop/update;
  không gồm validation, test, ghi checkpoint hoặc vẽ. Validation có cột thời gian riêng.
  Đo trên cùng máy, cùng device; có thể chạy lặp nhiều seed nếu cần báo cáo độ biến thiên.
- Seed giúp tái lập; không bảo đảm bitwise giống nhau giữa mọi thiết bị/phiên bản Torch.
- Không có kết quả MNIST được bịa sẵn trong gói. Bạn chạy lệnh để tạo số liệu thật.

## Kiểm tra đã thực hiện khi bàn giao

Đã kiểm tra cú pháp, đối chiếu gradient Softmax thủ công với autograd,
kiểm tra logits có độ lớn cao, batch cuối không đủ kích thước, và xác nhận evaluate
không sửa trọng số. Đã chạy toàn bộ luồng của hai model với dữ liệu giả nhỏ,
kiểm tra xuất biểu đồ và từ chối so sánh khi split khác nhau.
Chưa huấn luyện toàn bộ MNIST; kết quả kiểm thử giả không được đưa vào gói bàn giao.
