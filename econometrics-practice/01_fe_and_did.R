# ============================================================
# 计量入门练习 01：固定效应与双重差分（DiD）
# ============================================================
# 数据：世界银行 WDI，国家-年份面板
# 第一部分：贸易开放度与人均收入，比较混合 OLS 和固定效应
# 第二部分：2004 年欧盟东扩对中东欧 8 国出口的影响（DiD）
#
# 使用方法：
#   1. 在 RStudio 里把工作目录设为 econometrics-practice/ 文件夹
#      （菜单 Session > Set Working Directory > To Source File Location）
#   2. 第一次使用先安装包：
#      install.packages(c("fixest", "dplyr", "ggplot2"))
#   3. 不要一次跑完整个文件。一段一段地运行（选中后按 Ctrl/Cmd + Enter），
#      看到"思考题"就停下来，先自己回答，再往下跑。
#      参考答案在 answers.md。
# ============================================================

library(fixest)   # 面板回归：固定效应、聚类标准误、事件研究
library(dplyr)    # 数据整理
library(ggplot2)  # 画图

if (!file.exists("data/wdi_panel.csv")) {
  stop("找不到 data/wdi_panel.csv。请先把工作目录设为 econometrics-practice/ 文件夹。")
}
dir.create("output", showWarnings = FALSE)

wdi <- read.csv("data/wdi_panel.csv")

# 先看看数据长什么样。面板数据的一行 = 一个国家的一年。
head(wdi)
length(unique(wdi$iso3))   # 多少个国家
range(wdi$year)            # 年份范围


# ============================================================
# 第一部分：固定效应（Fixed Effects）
# ============================================================
# 研究问题：贸易开放度更高的国家，人均收入是不是更高？
#
# 变量都取对数，这样系数可以读作"弹性"：
# 贸易开放度每高 1%，人均 GDP 平均高多少 %。

fe_data <- wdi %>%
  filter(year >= 1995, year <= 2019,
         !is.na(gdppc), !is.na(trade_gdp), trade_gdp > 0) %>%   # 取对数前去掉 0 和缺失值
  mutate(log_gdppc = log(gdppc),
         log_trade = log(trade_gdp))

# ---- 模型 1：混合 OLS（pooled OLS）----
# 把所有国家、所有年份的观测混在一起跑回归。
# 它比较的主要是"国家之间"的差异：开放的国家和封闭的国家比。
#
# cluster = ~iso3 表示按国家聚类标准误：同一个国家不同年份的观测
# 并不独立（今年的波兰和去年的波兰很像），不聚类会低估不确定性。
m1 <- feols(log_gdppc ~ log_trade, data = fe_data, cluster = ~iso3)

# ---- 模型 2：国家固定效应 ----
# "| iso3" 给每个国家一个自己的截距。
# 效果是：只用"同一个国家内部、随时间的变化"来估计系数。
# 地理位置、国土面积、殖民历史这类不随时间变化的因素，全部被吸收掉。
m2 <- feols(log_gdppc ~ log_trade | iso3, data = fe_data, cluster = ~iso3)

# ---- 模型 3：双向固定效应（国家 + 年份）----
# 再加上年份固定效应，吸收所有国家共同经历的冲击，
# 比如全球化浪潮、2008 年金融危机。
m3 <- feols(log_gdppc ~ log_trade | iso3 + year, data = fe_data, cluster = ~iso3)

etable(m1, m2, m3)

# 思考题 1.1：系数从模型 1 到模型 3 变化很大。为什么会这样？
#             提示：想想新加坡、卢森堡这类国家，以及"所有国家都在同时变得更开放、更富"。
# 思考题 1.2：模型 3 的系数能解释成"扩大开放会让一个国家变富"吗？
#             列出至少两个理由说明为什么不能。
# 思考题 1.3：模型 2、3 的 R2 很高，但 Within R2 很低。这两个数分别是什么意思？
#             R2 高能说明模型"好"吗？


# ============================================================
# 第二部分：双重差分（Difference-in-Differences）
# ============================================================
# 研究问题：2004 年加入欧盟，有没有让中东欧国家的出口增加？
#
# 处理组：2004 年 5 月入盟的中东欧 8 国
# 对照组：同样经历了转型、但 2007 年前没有入盟的后社会主义国家
# 时间窗口：1995–2007 年
#   - 2007 年保加利亚、罗马尼亚入盟，2008 年全球金融危机，所以截止到 2007 年
#   - 保加利亚、罗马尼亚、黑山（数据从 2000 年才有）不放进样本

treated_countries <- c("CZE", "EST", "HUN", "LVA", "LTU", "POL", "SVK", "SVN")
control_countries <- c("ALB", "ARM", "BIH", "BLR", "GEO", "HRV",
                       "MDA", "MKD", "RUS", "SRB", "UKR")

did_data <- wdi %>%
  filter(iso3 %in% c(treated_countries, control_countries),
         year >= 1995, year <= 2007) %>%
  mutate(treated = as.integer(iso3 %in% treated_countries),  # 1 = 处理组
         post    = as.integer(year >= 2004),                 # 1 = 入盟之后
         log_gdppc = log(gdppc))

# 检查样本：每个国家应该有 13 年
table(did_data$treated, did_data$year)

# ---- 第 1 步：先画图 ----
# 做 DiD 之前一定要先看两组的走势。
# 实线是 2004 年入盟；虚线是 1998 年，入盟谈判开始的年份。
trend <- did_data %>%
  group_by(group = ifelse(treated == 1, "EU 2004 entrants", "Control"), year) %>%
  summarise(exports_gdp = mean(exports_gdp, na.rm = TRUE), .groups = "drop")

p <- ggplot(trend, aes(year, exports_gdp, colour = group)) +
  geom_line(linewidth = 1) +
  geom_point() +
  geom_vline(xintercept = 2003.5) +
  geom_vline(xintercept = 1998, linetype = "dashed") +
  scale_x_continuous(breaks = seq(1995, 2007, 2)) +
  labs(x = NULL, y = "Exports (% of GDP), group mean", colour = NULL,
       title = "Exports: EU 2004 entrants vs. control group") +
  theme_minimal() +
  theme(legend.position = "bottom")
print(p)
ggsave("output/did_trends.png", p, width = 7, height = 4.5, dpi = 150)

# 思考题 2.1：入盟前，两组的出口水平差很多（大约 46% 对 32%）。
#             这会破坏 DiD 吗？DiD 需要的到底是什么条件？

# ---- 第 2 步：手算 2×2 DiD ----
# DiD = (处理组 后 - 处理组 前) - (对照组 后 - 对照组 前)
# 第一个差：处理组自己前后的变化
# 第二个差：用对照组的变化代表"如果没入盟，处理组本来会怎么变"
means <- did_data %>%
  group_by(treated, post) %>%
  summarise(exports_gdp = mean(exports_gdp, na.rm = TRUE), .groups = "drop")
means

m <- function(t, p) means$exports_gdp[means$treated == t & means$post == p]
did_by_hand <- (m(1, 1) - m(1, 0)) - (m(0, 1) - m(0, 0))
did_by_hand

# ---- 第 3 步：用回归做同样的事 ----
# treated:post 是交互项，只有"处理组且入盟之后"取 1。
# 国家固定效应吸收了 treated，年份固定效应吸收了 post，所以它们不用单独放。
did_reg <- feols(exports_gdp ~ treated:post | iso3 + year,
                 data = did_data, cluster = ~iso3)
summary(did_reg)

# 思考题 2.2：回归系数和手算的结果一样吗？为什么一样？
#             回归的好处是什么？（提示：标准误、加控制变量）

# ---- 第 4 步：事件研究（event study）----
# 不再只分"前"和"后"，而是估计每一年的处理效应，以 2003 年为基准（设为 0）。
# 入盟前各年的系数可以用来检查"平行趋势"：如果入盟前两组就在分化，
# 这些系数会明显偏离 0。
event_study <- feols(exports_gdp ~ i(year, treated, ref = 2003) | iso3 + year,
                     data = did_data, cluster = ~iso3)
summary(event_study)

iplot(event_study,
      main = "Event study: EU accession and exports (% of GDP)",
      xlab = "Year (reference = 2003)")
png("output/event_study.png", width = 1050, height = 675, res = 150)
iplot(event_study,
      main = "Event study: EU accession and exports (% of GDP)",
      xlab = "Year (reference = 2003)")
invisible(dev.off())

# 思考题 2.3：入盟前（1995–2002）的系数大体在 0 附近。这能"证明"平行趋势成立吗？
# 思考题 2.4：效应在 2004 年很小，到 2006–2007 年才变大。可能的解释有哪些？
# 思考题 2.5：这个 DiD 的识别面临哪些威胁？从区域国别的知识出发想一想：
#             - 谁能入盟是随机的吗？
#             - 入盟谈判 1998 年就开始了，这对"前/后"的划分意味着什么？
#             - 对照组里的俄罗斯、乌克兰，2000 年代经历了什么？
#             - 1999 年科索沃战争影响了对照组里的哪些国家？
# 思考题 2.6：样本只有 19 个国家，按国家聚类意味着只有 19 个"簇"。
#             这对标准误有什么影响？


# ============================================================
# 第三部分：自己动手
# ============================================================
# 下面的练习需要你自己改代码。卡住了可以问 Claude，
# 但建议先让它解释思路，而不是直接给出代码。

# 练习 3.1：换结果变量。把 exports_gdp 换成 fdi_gdp（FDI 净流入占 GDP 的 %）
#           和 log_gdppc（人均 GDP 的对数），重新跑第 3、4 步。
#           结果有什么不同？FDI 的事件研究图为什么这么"乱"？

# 练习 3.2：稳健性检验。从对照组里去掉 RUS、UKR、BLR，或者去掉受科索沃战争
#           影响的 ALB、MKD、SRB，结果变了多少？
#           提示：did_data %>% filter(!iso3 %in% c(...))

# 练习 3.3：安慰剂检验（placebo test）。只保留 1995–2003 年的数据，
#           假装入盟发生在 2000 年（post = year >= 2000），再跑一次 DiD。
#           如果设计没问题，你预期会看到什么？实际看到了什么？

# 练习 3.4（进阶）：把保加利亚、罗马尼亚（2007 年入盟）也作为处理组加进来，
#           时间窗口延长到 2010 年。这时不同国家的处理时间不一样，
#           叫"交错 DiD"（staggered DiD）。读一读 .claude/skills/econ-write/
#           identification-strategies.md 里关于交错 DiD 的部分，
#           再试试 fixest 的 sunab() 函数。
