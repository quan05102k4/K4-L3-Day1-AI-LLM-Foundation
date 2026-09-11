# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Ở temperature 0.0 và 0.5, hai lần gọi lặp lại đều hội tụ về cùng một chủ đề (cà phê Việt Nam), chỉ khác nhau chút ít cách diễn đạt — model ưu tiên câu trả lời có xác suất cao nhất. Từ 1.0 trở lên, hai lần gọi bắt đầu chọn những sự thật hoàn toàn khác nhau (cà phê so với Vịnh Hạ Long), và ở 1.5 thì khác biệt rõ nhất (Vịnh Hạ Long so với hang Sơn Đoòng). Quy luật: temperature càng cao, phân phối xác suất càng được "làm phẳng" nên model càng dễ chọn các phương án ít phổ biến hơn — output đa dạng, sáng tạo hơn nhưng cũng kém ổn định/khó đoán hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Khoảng 0.0–0.3. Chatbot hỗ trợ khách hàng cần trả lời chính xác, nhất quán về chính sách/thông tin sản phẩm hơn là sáng tạo — temperature thấp giảm rủi ro bịa thông tin (hallucination) và giúp câu trả lời có thể tái lập, dễ kiểm thử/QA hơn so với temperature cao.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Workload: 10.000 người dùng × 3 lượt/ngày × 350 token output = 10.500.000 token output/ngày (10.500 nghìn-token). Theo bảng giá output trong `template.py` (GPT-4o: $0.010/1K, GPT-4o-mini: $0.0006/1K): GPT-4o ≈ 10.500 × 0.010 = **$105/ngày**, GPT-4o-mini ≈ 10.500 × 0.0006 = **$6.30/ngày** → GPT-4o đắt hơn khoảng **16.7 lần** (đúng bằng tỷ lệ đơn giá output của hai model). GPT-4o xứng đáng khi tác vụ cần suy luận phức tạp/độ chính xác cao (tư vấn pháp lý, debug code, phân tích số liệu); GPT-4o-mini phù hợp cho khối lượng lớn tác vụ đơn giản (FAQ, phân loại intent, tóm tắt ngắn) nơi chênh lệch chất lượng không đáng để trả thêm 16.7 lần chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Gọi thật `chat_with_system_prompt` với cùng câu hỏi: bản "giáo viên tiểu học" trả lời 186 từ (859 ký tự), dùng ví dụ đời thường (so sánh blockchain với "cuốn sổ nhật ký chung của lớp", "trang" = khối, không có thuật ngữ tiếng Anh); bản "chuyên gia tài chính" trả lời 182 từ nhưng dài hơn về ký tự (1004), trình bày bằng bảng và thuật ngữ kỹ thuật (distributed ledger, cryptographic hash, Merkle root, proof-of-work, SHA-256). Độ dài tính theo số từ gần như nhau nhưng mật độ thông tin và độ phức tạp từ vựng khác hẳn — persona không đổi số lượng nội dung nhiều bằng đổi **cách diễn đạt, mức độ trừu tượng và định dạng trình bày**. Điều này cho thấy system prompt quyết định "giọng điệu" và đối tượng người đọc mà model nhắm tới, chứ không chỉ là một gợi ý phong cách hời hợt.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Đoạn văn tiếng Việt thử nghiệm dài 103 từ. `count_tokens(..., model="gpt-4o")` (tiktoken thật) đếm được **131 token**, trong khi ước lượng `số_từ / 0.75` cho ra **137** — chênh nhau 6 token (~4.6%), khá gần nhau ở mẫu này. Để kiểm tra rõ hơn phần "vì sao tiếng Việt tốn token hơn tiếng Anh", tôi dịch cùng nội dung sang tiếng Anh (77 từ) và đếm token cả hai: tiếng Việt ra tỷ lệ **1.27 token/từ**, tiếng Anh chỉ **1.13 token/từ** — tiếng Việt tốn nhiều hơn thật. Lý do: bộ mã hoá BPE của tiktoken được huấn luyện chủ yếu trên dữ liệu tiếng Anh, nên các từ tiếng Anh phổ biến thường gộp vừa một token; còn tiếng Việt có dấu (ký tự Unicode tổ hợp) và nhiều âm tiết ít xuất hiện trong dữ liệu huấn luyện, nên hay bị tách thành 2 token trở lên cho mỗi từ/âm tiết.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Ở Block 1, `compare_models` đo được `gpt4o_latency` thật ≈ 2.1 giây cho một câu trả lời từ model đang dùng trong lab này (`openai/gpt-oss-120b` qua Groq, cấu hình trong `.env`, không phải OpenAI GPT-4o thật) — nếu chờ trọn vẹn response rồi mới hiển thị, người dùng nhìn màn hình trống hơn 2 giây trước khi thấy chữ đầu tiên, cảm giác "đơ"/chậm. Streaming quan trọng nhất trong các ứng dụng chat tương tác trực tiếp (chatbot, trợ lý ảo) vì nó hiển thị chữ ngay khi model sinh ra, giảm cảm giác chờ đợi dù tổng thời gian xử lý không đổi. Ngược lại, non-streaming phù hợp hơn khi hệ thống cần xử lý toàn bộ output trước khi dùng — ví dụ gọi API để lấy JSON có cấu trúc rồi parse, chạy batch job không có người xem trực tiếp, hoặc khi cần kiểm duyệt/validate nội dung trước khi trả về cho người dùng (không thể kiểm duyệt từng chunk nhỏ một cách đáng tin cậy).

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Đo thật `retry_with_backoff(base_delay=0.1)` khi hàm luôn lỗi 3 lần đầu: khoảng cách giữa các lần thử là **0.114s → 0.201s → 0.406s** — đúng cấp số nhân (nhân đôi mỗi lần), không phải delay cố định. Lợi thế: nếu server đang quá tải, chờ ngày càng lâu cho mỗi lần thử tiếp theo giúp server có thời gian hồi phục thay vì bị dội thêm request ngay lập tức, đồng thời tổng thời gian client "kiên nhẫn" tăng dần thay vì bỏ cuộc/spam liên tục. Nếu hàng nghìn client cùng dùng delay cố định giống nhau (ví dụ luôn chờ đúng 1 giây), chúng có xu hướng đồng bộ hoá và cùng retry lại vào đúng cùng một thời điểm — tạo thành đợt sóng request dồn dập (thundering herd), khiến server vừa mới hồi phục lại lập tức bị quá tải lần nữa. Thực tế nên kết hợp thêm "jitter" (làm nhiễu ngẫu nhiên nhỏ vào delay) để tránh việc các client vẫn vô tình đồng bộ dù đã dùng backoff.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona tôi dùng khi chạy demo thật: **"Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt."** Hai lựa chọn từ ngữ quan trọng: (1) "**trả lời ngắn gọn**" — vì `run_assistant` stream từng chunk qua CLI và độ trễ mỗi lượt đo thật ở Block 1 đã ~2 giây, câu trả lời dài vừa làm người dùng chờ lâu hơn vừa tốn thêm token/chi phí (`estimate_cost`); (2) "**bằng tiếng Việt**" — chỉ định rõ ngôn ngữ để model không tự chuyển sang tiếng Anh khi câu hỏi ngắn/mơ hồ, đảm bảo trải nghiệm nhất quán cho học viên Việt Nam. Từ "thân thiện" đặt tông giọng gần gũi, khuyến khích học viên hỏi thoải mái thay vì cảm thấy đang bị một hệ thống trang trọng "chấm điểm".

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: `history` chỉ giữ 3 lượt gần nhất (`history[-6:]`) và **không có bộ nhớ giữa các phiên** — khi test thật, sau lượt 2 (`history` dài 4 message) mọi ngữ cảnh cũ hơn đã mất, và nếu tắt chương trình rồi mở lại thì trợ lý hoàn toàn "quên" người dùng là ai, dù mới trò chuyện trước đó vài phút. Cải thiện cụ thể: lưu `history` (hoặc một bản tóm tắt ngắn của các lượt bị cắt) ra một file JSON theo phiên, ví dụ `sessions/<user_id>.json`; khi `run_assistant` khởi động, đọc file này để nạp lại vài lượt gần nhất trước khi vào vòng lặp, và ghi đè lại file sau mỗi lượt (`json.dump(history, f)`). Cách này giữ được trí nhớ xuyên phiên mà không cần thay đổi cách cắt `history` trong một phiên đang chạy.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
