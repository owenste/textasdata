# 海外博士论文元数据抓取脚本

- `sets.py <OAI地址> <正则>`：列出机构知识库中名称匹配的集合（set）。
  例：`python3 sets.py http://etheses.lse.ac.uk/cgi/oai2 "international|thesis"`
- `harvest.py <OAI地址> <setSpec或-> <输出.json> [起始日期]`：按 oai_dc 批量抓取题名、作者、日期、摘要、主题。
  例：`python3 harvest.py https://cadmus.eui.eu/oai/request col_1814_4857 eui.json 2022-01-01`
- `dspace7.py <站点> <检索式> <输出.json>`：调用 DSpace 7 检索接口（如 MIT）。

注意：OAI 的 from 参数是记录修改时间而非学位年份，需按 dc:date 再筛选；请控制访问频率并遵守各站点条款。
已验证可用：哈佛DASH、耶鲁EliScholar、MIT DSpace、LSE Theses Online、EUI Cadmus、剑桥Apollo、杜克、宾大、BU。
