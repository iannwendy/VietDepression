# Appendix A: Complete Keyword Lists

This appendix documents all Vietnamese-language keywords used in the project for two purposes: (1) YouTube video search during crawling, and (2) weak-label scoring of collected comments.

---

## A.1 YouTube Search Keywords (`DEFAULT_KEYWORDS`)

These 264 unique keywords are used to search for relevant YouTube videos. They cover a broad range of depression, anxiety, loneliness, academic/work stress, heartbreak, and mental health topics in Vietnamese. Note: entries 265-275 in the table below are duplicates of earlier keywords and should be disregarded; the authoritative count is 264 unique keywords from `keywords.py`.

| # | Keyword (Vietnamese) |
|---|----------------------|
| 1 | trầm cảm |
| 2 | tôi bị trầm cảm |
| 3 | dấu hiệu trầm cảm |
| 4 | vượt qua trầm cảm |
| 5 | sống với trầm cảm |
| 6 | cô đơn |
| 7 | cảm thấy cô đơn |
| 8 | nỗi cô đơn |
| 9 | tâm sự đêm khuya |
| 10 | tâm sự buồn |
| 11 | mệt mỏi cuộc sống |
| 12 | chán nản cuộc sống |
| 13 | không còn động lực |
| 14 | mất động lực học tập |
| 15 | áp lực học tập |
| 16 | stress học đường |
| 17 | áp lực thi cử |
| 18 | áp lực điểm số |
| 19 | lo âu |
| 20 | rối loạn lo âu |
| 21 | căng thẳng kéo dài |
| 22 | stress công việc |
| 23 | áp lực công việc |
| 24 | burnout |
| 25 | kiệt sức công việc |
| 26 | mất ngủ |
| 27 | mất ngủ vì stress |
| 28 | suy nghĩ tiêu cực |
| 29 | tuyệt vọng |
| 30 | muốn biến mất |
| 31 | không ai hiểu mình |
| 32 | cảm giác trống rỗng |
| 33 | khóc một mình |
| 34 | thất tình |
| 35 | đau khổ sau chia tay |
| 36 | buồn vì thất tình |
| 37 | áp lực gia đình |
| 38 | mâu thuẫn gia đình |
| 39 | bị bố mẹ tạo áp lực |
| 40 | không được công nhận |
| 41 | tự ti bản thân |
| 42 | ghét bản thân |
| 43 | mất niềm tin |
| 44 | không muốn nói chuyện với ai |
| 45 | sống vô nghĩa |
| 46 | cần người lắng nghe |
| 47 | tâm sự tuổi trẻ |
| 48 | chữa lành nỗi đau |
| 49 | vượt qua áp lực |
| 50 | sức khỏe tinh thần |
| 51 | tôi thấy cô đơn |
| 52 | tôi thấy mệt mỏi |
| 53 | tôi không ổn |
| 54 | mình không ổn |
| 55 | tôi kiệt sức |
| 56 | mình kiệt sức |
| 57 | tôi mất ngủ |
| 58 | đêm không ngủ được |
| 59 | khó ngủ |
| 60 | rối loạn giấc ngủ |
| 61 | mất ngủ kéo dài |
| 62 | thức trắng đêm |
| 63 | overthinking ban đêm |
| 64 | suy nghĩ quá nhiều |
| 65 | nghĩ nhiều mệt mỏi |
| 66 | tâm sự một mình |
| 67 | tâm sự với người lạ |
| 68 | tâm sự radio |
| 69 | podcast tâm sự |
| 70 | chuyện buồn cuộc sống |
| 71 | nỗi buồn không tên |
| 72 | nhạc buồn tâm trạng |
| 73 | nhạc chữa lành |
| 74 | nhạc cho người cô đơn |
| 75 | nhạc thất tình |
| 76 | nhạc buồn đêm khuya |
| 77 | mưa buồn tâm trạng |
| 78 | tâm trạng buồn |
| 79 | buồn không lý do |
| 80 | cảm xúc tiêu cực |
| 81 | năng lượng tiêu cực |
| 82 | cạn năng lượng |
| 83 | kiệt quệ tinh thần |
| 84 | suy sụp tinh thần |
| 85 | khủng hoảng tinh thần |
| 86 | khủng hoảng tuổi 20 |
| 87 | khủng hoảng tuổi trẻ |
| 88 | mất phương hướng |
| 89 | lạc lõng |
| 90 | cảm thấy lạc lõng |
| 91 | không có ai bên cạnh |
| 92 | không ai lắng nghe |
| 93 | không ai quan tâm |
| 94 | không ai thương mình |
| 95 | không ai cần mình |
| 96 | không muốn gặp ai |
| 97 | không muốn ra ngoài |
| 98 | thu mình lại |
| 99 | sợ giao tiếp |
| 100 | ngại giao tiếp |
| 101 | lo âu xã hội |
| 102 | rối loạn hoảng sợ |
| 103 | cơn hoảng loạn |
| 104 | panic attack |
| 105 | bất an |
| 106 | luôn thấy bất an |
| 107 | sợ hãi vô cớ |
| 108 | mệt mỏi vì gia đình |
| 109 | gia đình không hiểu mình |
| 110 | bố mẹ không hiểu con |
| 111 | áp lực từ bố mẹ |
| 112 | áp lực làm con ngoan |
| 113 | áp lực kỳ vọng gia đình |
| 114 | bị so sánh với người khác |
| 115 | bố mẹ so sánh |
| 116 | mâu thuẫn với bố mẹ |
| 117 | cãi nhau với gia đình |
| 118 | tổn thương gia đình |
| 119 | áp lực tài chính |
| 120 | áp lực tiền bạc |
| 121 | nợ nần áp lực |
| 122 | thất nghiệp áp lực |
| 123 | mất việc |
| 124 | chán việc |
| 125 | kiệt sức vì công việc |
| 126 | mệt mỏi công sở |
| 127 | toxic workplace |
| 128 | môi trường làm việc độc hại |
| 129 | sếp toxic |
| 130 | áp lực deadline |
| 131 | deadline dí |
| 132 | burnout công sở |
| 133 | chán đi làm |
| 134 | muốn nghỉ việc |
| 135 | mất động lực đi làm |
| 136 | áp lực đại học |
| 137 | áp lực thi đại học |
| 138 | áp lực học sinh |
| 139 | áp lực sinh viên |
| 140 | stress sinh viên |
| 141 | stress mùa thi |
| 142 | áp lực điểm thi |
| 143 | sợ rớt môn |
| 144 | rớt môn |
| 145 | chán học |
| 146 | mất động lực học |
| 147 | không muốn đi học |
| 148 | bị bắt nạt |
| 149 | bạo lực học đường |
| 150 | cô lập ở trường |
| 151 | không có bạn bè |
| 152 | tự ti ngoại hình |
| 153 | tự ti học tập |
| 154 | tự ti giao tiếp |
| 155 | ghét chính mình |
| 156 | không chấp nhận bản thân |
| 157 | cảm thấy vô dụng |
| 158 | tôi vô dụng |
| 159 | mình vô dụng |
| 160 | không làm được gì |
| 161 | luôn thất bại |
| 162 | sợ thất bại |
| 163 | mất hy vọng |
| 164 | không còn hy vọng |
| 165 | không còn tương lai |
| 166 | tương lai mịt mờ |
| 167 | không biết sống để làm gì |
| 168 | sống để làm gì |
| 169 | đời vô nghĩa |
| 170 | cuộc đời vô nghĩa |
| 171 | muốn buông xuôi |
| 172 | buông xuôi tất cả |
| 173 | không chịu nổi nữa |
| 174 | không thể chịu nổi |
| 175 | muốn ngủ mãi |
| 176 | chỉ muốn ngủ mãi |
| 177 | muốn tan biến |
| 178 | không muốn tồn tại |
| 179 | không thiết sống |
| 180 | không còn lý do sống |
| 181 | muốn chết |
| 182 | nghĩ đến cái chết |
| 183 | ý định tự tử |
| 184 | tự tử |
| 185 | đừng tự tử |
| 186 | vượt qua ý định tự tử |
| 187 | phòng chống tự tử |
| 188 | dấu hiệu muốn tự tử |
| 189 | cứu tôi với |
| 190 | ai cứu tôi với |
| 191 | cần được giúp đỡ |
| 192 | cần ai đó nói chuyện |
| 193 | cần người tâm sự |
| 194 | cần người chia sẻ |
| 195 | xin lời khuyên tâm lý |
| 196 | tư vấn tâm lý |
| 197 | trị liệu tâm lý |
| 198 | đi khám tâm lý |
| 199 | gặp bác sĩ tâm lý |
| 200 | chuyên gia tâm lý |
| 201 | rối loạn cảm xúc |
| 202 | rối loạn lưỡng cực |
| 203 | rối loạn stress sau sang chấn |
| 204 | sang chấn tâm lý |
| 205 | tổn thương tâm lý |
| 206 | chữa lành tâm lý |
| 207 | chữa lành đứa trẻ bên trong |
| 208 | inner child |
| 209 | yêu bản thân |
| 210 | học cách yêu bản thân |
| 211 | tha thứ cho bản thân |
| 212 | vượt qua nỗi đau |
| 213 | vượt qua cô đơn |
| 214 | vượt qua thất tình |
| 215 | vượt qua chia tay |
| 216 | sau chia tay |
| 217 | người yêu cũ |
| 218 | bị phản bội |
| 219 | bị bỏ rơi |
| 220 | cảm giác bị bỏ rơi |
| 221 | lụy tình |
| 222 | đau sau chia tay |
| 223 | không quên được người yêu cũ |
| 224 | tình yêu toxic |
| 225 | mối quan hệ độc hại |
| 226 | gaslighting |
| 227 | bị thao túng tâm lý |
| 228 | tổn thương trong tình yêu |
| 229 | chán đời tuổi trẻ |
| 230 | áp lực tuổi 20 |
| 231 | áp lực tuổi 25 |
| 232 | khủng hoảng hiện sinh |
| 233 | existential crisis |
| 234 | cảm giác trống rỗng trong lòng |
| 235 | đêm khuya suy nghĩ |
| 236 | một mình trong phòng |
| 237 | khóc trong im lặng |
| 238 | nước mắt đêm khuya |
| 239 | mệt mỏi nhưng không nói ra |
| 240 | nụ cười che giấu nỗi buồn |
| 241 | ẩn sau nụ cười |
| 242 | tâm sự người trẻ |
| 243 | người trẻ trầm cảm |
| 244 | sinh viên trầm cảm |
| 245 | học sinh trầm cảm |
| 246 | nhân viên văn phòng trầm cảm |
| 247 | mẹ bỉm trầm cảm |
| 248 | trầm cảm sau sinh |
| 249 | trầm cảm tuổi dậy thì |
| 250 | trầm cảm ở người trẻ |
| 251 | dấu hiệu lo âu |
| 252 | dấu hiệu stress |
| 253 | dấu hiệu burnout |
| 254 | dấu hiệu kiệt sức |
| 255 | cách vượt qua trầm cảm |
| 256 | cách giảm stress |
| 257 | cách ngủ ngon |
| 258 | cách hết overthinking |
| 259 | cách vượt qua cô đơn |
| 260 | cách lấy lại động lực |
| 261 | làm sao để có động lực |
| 262 | làm sao để bớt buồn |
| 263 | làm sao để hết mệt mỏi |
| 264 | làm sao để yêu bản thân |
| 265 | yêu bản thân |
| 266 | tự ti bản thân |
| 267 | không còn động lực |
| 268 | kiệt sức |
| 269 | mất ngủ |
| 270 | áp lực |
| 271 | cô đơn |
| 272 | burnout |
| 273 | overthinking |
| 274 | chán nản |
| 275 | mệt mỏi |

**Total: 264 unique keywords** (from `DEFAULT_KEYWORDS` in `keywords.py`)

---

## A.2 Weak-Label Scoring Keywords (`label_config.py`)

These keywords are used to score each crawled comment for weak-label assignment. Each category has an associated weight used in the scoring formula:

```
depression_score = (strong_count × 5) + (medium_count × 3) + (normal_count × −2)
```

### A.2.1 Depression Strong Keywords (Weight: +5)

These keywords indicate severe depression or suicidal ideation. Each match adds 5 points to the depression score.

| # | Keyword (Vietnamese) | Romanization |
|---|----------------------|--------------|
| 1 | muốn chết | muon chet |
| 2 | tự tử | tu tu |
| 3 | không muốn sống | khong muon song |
| 4 | không thiết sống | khong thiet song |
| 5 | muốn biến mất | muon bien mat |
| 6 | muốn tan biến | muon tan bien |
| 7 | muốn ngủ mãi | muon ngu mai |
| 8 | kết thúc cuộc đời | ket thuc cuoc doi |
| 9 | chấm dứt cuộc đời | cham dut cuoc doi |
| 10 | không còn lý do sống | khong con ly do song |
| 11 | sống vô nghĩa | song vo nghia |
| 12 | cuộc sống vô nghĩa | cuoc song vo nghia |
| 13 | vô nghĩa | vo nghia |
| 14 | tuyệt vọng | tuyet vong |
| 15 | trầm cảm | tram cam |
| 16 | rối loạn trầm cảm | roi loan tram cam |
| 17 | đau khổ quá | dau kho qua |
| 18 | không chịu nổi nữa | khong chiu noi nua |
| 19 | không thể chịu nổi | khong the chiu noi |
| 20 | chỉ muốn buông xuôi | chi muon buong xuoi |
| 21 | buông xuôi tất cả | buong xuoi tat ca |
| 22 | không còn hy vọng | khong con hy vong |
| 23 | không còn tương lai | khong con tuong lai |
| 24 | chẳng còn gì để mất | chang con gi de mat |

**Total: 24 keywords (12 Vietnamese + 12 romanization variants)**

---

### A.2.2 Depression Medium Keywords (Weight: +3)

These keywords indicate moderate depression symptoms. Each match adds 3 points to the depression score.

| # | Keyword (Vietnamese) | Romanization |
|---|----------------------|--------------|
| 1 | cô đơn | co don |
| 2 | mệt mỏi | met moi |
| 3 | áp lực | ap luc |
| 4 | stress | stress |
| 5 | mất ngủ | mat ngu |
| 6 | kiệt sức | kiet suc |
| 7 | burnout | burnout |
| 8 | không còn động lực | khong con dong luc |
| 9 | mất động lực | mat dong luc |
| 10 | chán đời | chan doi |
| 11 | chán nản | chan nan |
| 12 | trống rỗng | trong rong |
| 13 | suy sụp | suy sup |
| 14 | bất lực | bat luc |
| 15 | lo âu | lo au |
| 16 | overthinking | overthinking |
| 17 | khóc một mình | khoc mot minh |
| 18 | không ai hiểu | khong ai hieu |
| 19 | áp lực gia đình | ap luc gia dinh |
| 20 | áp lực học tập | ap luc hoc tap |
| 21 | áp lực công việc | ap luc cong viec |
| 22 | mất phương hướng | mat phuong huong |
| 23 | tự ti | tu ti |
| 24 | ghét bản thân | ghet ban than |
| 25 | mất niềm tin | mat niem tin |
| 26 | không muốn nói chuyện | khong muon noi chuyen |
| 27 | cạn năng lượng | can nang luong |
| 28 | mệt quá | met qua |
| 29 | buồn quá | buon qua |
| 30 | đau lòng | dau long |
| 31 | tổn thương | ton thuong |

**Total: 31 keywords (15 Vietnamese + 15 romanization variants + 1 English)**

---

### A.2.3 Normal Keywords (Weight: -2)

These positive/negative keywords reduce the depression score. Each match subtracts 2 points from the depression score.

| # | Keyword (Vietnamese) | English Translation |
|---|----------------------|---------------------|
| 1 | video hay | good video |
| 2 | hay quá | so good |
| 3 | hay qua | so good |
| 4 | cảm ơn | thank you |
| 5 | cam on | thank you |
| 6 | vui quá | so happy |
| 7 | vui qua | so happy |
| 8 | bổ ích | informative |
| 9 | bo ich | informative |
| 10 | ủng hộ | support |
| 11 | ung ho | support |
| 12 | hài quá | so funny |
| 13 | hai qua | so funny |
| 14 | thích quá | love it |
| 15 | thich qua | love it |
| 16 | tuyệt vời | wonderful |
| 17 | tuyet voi | wonderful |
| 18 | ý nghĩa quá | so meaningful |
| 19 | y nghia qua | so meaningful |
| 20 | cười xỉu | laughing so hard |
| 21 | cuoi xiu | laughing so hard |
| 22 | giải trí | entertainment |
| 23 | giai tri | entertainment |
| 24 | dễ thương | cute |
| 25 | de thuong | cute |
| 26 | đáng yêu | adorable |
| 27 | dang yeu | adorable |
| 28 | học được nhiều | learned a lot |
| 29 | hoc duoc nhieu | learned a lot |
| 30 | cảm động quá | so touching |
| 31 | cam dong qua | so touching |
| 32 | xem cuốn | entertaining |
| 33 | xem cuon | entertaining |
| 34 | quá đỉnh | so great |
| 35 | qua dinh | so great |
| 36 | đỉnh quá | so great |
| 37 | dinh qua | so great |
| 38 | rất hay | very good |
| 39 | rat hay | very good |
| 40 | thật ý nghĩa | truly meaningful |
| 41 | that y nghia | truly meaningful |
| 42 | chúc mừng | congratulations |
| 43 | chuc mung | congratulations |
| 44 | iu quá | love so much |
| 45 | iu qua | love so much |
| 46 | yêu quá | love so much |
| 47 | yeu qua | love so much |

**Total: 47 keywords (24 Vietnamese + 23 romanization variants)**

---

### A.2.4 Review Context Keywords (Flag Only)

These keywords do not affect the depression score directly but flag the comment for human review because they may cause context ambiguity.

| # | Keyword (Vietnamese) | Romanization | Note |
|---|----------------------|--------------|------|
| 1 | đừng tự tử | dung tu tu | Anti-suicide prevention message |
| 2 | không nên tự tử | khong nen tu tu | Anti-suicide prevention message |
| 3 | đừng nghĩ quẩn | dung nghi quan | Encouragement not to despair |
| 4 | hãy đi khám | hay di kham | Seeking professional help |
| 5 | tìm chuyên gia | tim chuyen gia | Seeking professional help |
| 6 | gọi đường dây nóng | goi duong day nong | Crisis hotline reference |
| 7 | hãy chia sẻ | hay chia se | Encouragement to share |
| 8 | cố lên | co len | "Keep going" encouragement |
| 9 | mong bạn ổn | mong ban on | Wishing well for someone |
| 10 | chết cười | chet cuoi | "Dying of laughter" (idiom) |
| 11 | cười chết | cuoi chet | "Dying of laughter" (idiom) |

**Total: 11 keywords (11 Vietnamese + 11 romanization variants)**

---

## A.3 Keyword Scoring Configuration

The scoring thresholds from `config.py` are:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DEPRESSION_STRONG_WEIGHT` | 5 | Score added per strong keyword match |
| `DEPRESSION_MEDIUM_WEIGHT` | 3 | Score added per medium keyword match |
| `NORMAL_WEIGHT` | -2 | Score subtracted per normal keyword match |
| `DEPRESSION_AUTO_THRESHOLD` | 5 | Auto-label as depression if score >= this |
| `NORMAL_AUTO_THRESHOLD` | -2 | Auto-label as normal if score <= this |
| `HIGH_CONFIDENCE_DEPRESSION_THRESHOLD` | 8 | High confidence depression if score >= this |
| `HIGH_CONFIDENCE_NORMAL_THRESHOLD` | -4 | High confidence normal if score <= this |

---

## A.4 Summary Statistics

| Category | Count |
|----------|-------|
| YouTube Search Keywords | 275 |
| Depression Strong Keywords | 24 (12 Vietnamese + 12 romanization) |
| Depression Medium Keywords | 31 (15 Vietnamese + 15 romanization + 1 English) |
| Normal Keywords | 47 (24 Vietnamese + 23 romanization) |
| Review Context Keywords | 11 (11 Vietnamese + 11 romanization) |
