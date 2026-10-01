# ============================================================
# 从世界银行 WDI 下载练习用的国家-年份面板数据
# ============================================================
# 练习脚本默认直接读取已经下载好的 data/wdi_panel.csv，
# 一般不需要运行本文件。想更新数据，或想换别的指标时再运行。
#
# 运行方法：把工作目录设为 econometrics-practice/，然后
#   source("data/download_wdi.R")
#
# 世界银行 API 偶尔会超时或返回 400，所以下面每次请求都会自动重试。

library(jsonlite)
library(dplyr)
library(tidyr)

# 要下载的指标：左边是我们起的变量名，右边是 WDI 的指标代码。
# 想加别的指标，可以在 https://data.worldbank.org/indicator 查代码。
indicators <- c(
  exports_gdp  = "NE.EXP.GNFS.ZS",       # 货物和服务出口（占 GDP 的 %）
  fdi_gdp      = "BX.KLT.DINV.WD.GD.ZS", # 外国直接投资净流入（占 GDP 的 %）
  gdppc        = "NY.GDP.PCAP.KD",       # 人均 GDP（2015 年不变价美元）
  gdppc_growth = "NY.GDP.PCAP.KD.ZG",    # 人均 GDP 增长率（%）
  trade_gdp    = "NE.TRD.GNFS.ZS"        # 贸易开放度：进出口总额占 GDP 的 %
)
years <- "1990:2019"

# 带重试的请求函数
get_json <- function(url, tries = 5) {
  for (i in seq_len(tries)) {
    res <- tryCatch(fromJSON(url), error = function(e) NULL)
    if (!is.null(res) && length(res) == 2) return(res)
    message("请求失败，3 秒后重试（第 ", i, " 次）：", url)
    Sys.sleep(3)
  }
  stop("多次重试仍然失败，请稍后再试：", url)
}

# 1. 国家元数据：用来剔除"世界""欧元区"这类地区汇总行
meta <- get_json("https://api.worldbank.org/v2/country?format=json&per_page=400")[[2]]
meta <- data.frame(
  iso3    = meta$id,
  country = meta$name,
  region  = meta$region$value
) %>%
  filter(region != "Aggregates")

# 2. 逐个指标下载所有国家的数据（长表：一行 = 国家 × 年份 × 指标）
long <- bind_rows(lapply(names(indicators), function(nm) {
  url <- sprintf(
    "https://api.worldbank.org/v2/country/all/indicator/%s?format=json&date=%s&per_page=20000",
    indicators[[nm]], years
  )
  x <- get_json(url)[[2]]
  data.frame(iso3 = x$countryiso3code, year = as.integer(x$date), var = nm, value = x$value)
})) %>%
  filter(iso3 %in% meta$iso3)

# 3. 转成宽表：一行 = 一个国家的一年，这就是"面板数据"的标准格式
wdi_panel <- long %>%
  pivot_wider(names_from = var, values_from = value) %>%
  inner_join(meta, by = "iso3") %>%
  select(iso3, country, region, year, all_of(names(indicators))) %>%
  arrange(iso3, year)

write.csv(wdi_panel, "data/wdi_panel.csv", row.names = FALSE)
message("已保存 data/wdi_panel.csv：", nrow(wdi_panel), " 行，",
        n_distinct(wdi_panel$iso3), " 个国家")
