from datetime import date
from functools import lru_cache

# 法定节假日 - 转换为set以提高查找性能
holiday_date = []

# 2000年公众假日 (来源: 依据1999年9月18日《国务院关于修改<全国年节及纪念日放假办法>的决定》)
holiday_date.extend([
  # 元旦
  date(2000, 1, 1), date(2000, 1, 2), date(2000, 1, 3),
  # 春节
  date(2000, 2, 5), date(2000, 2, 6), date(2000, 2, 7), date(2000, 2, 8), date(2000, 2, 9), date(2000, 2, 10), date(2000, 2, 11),
  # 劳动节
  date(2000, 5, 1), date(2000, 5, 2), date(2000, 5, 3), date(2000, 5, 4), date(2000, 5, 5), date(2000, 5, 6), date(2000, 5, 7),
  # 国庆节
  date(2000, 10, 1), date(2000, 10, 2), date(2000, 10, 3), date(2000, 10, 4), date(2000, 10, 5), date(2000, 10, 6), date(2000, 10, 7)
])
# 2001年公众假日 (来源: https://zh.wikisource.org/wiki/%E5%9B%BD%E5%8A%A1%E9%99%A2%E5%8A%9E%E5%85%AC%E5%8E%85%E5%85%B3%E4%BA%8E2001%E5%B9%B4%E6%98%A5%E8%8A%82%E3%80%81%E2%80%9C%E4%BA%94%E4%B8%80%E2%80%9D%E3%80%81%E2%80%9C%E5%8D%81%E4%B8%80%E2%80%9D%E6%94%BE%E5%81%87%E5%AE%89%E6%8E%92%E7%9A%84%E9%80%9A%E7%9F%A5)
holiday_date.extend([
  # 元旦
  date(2001, 1, 1),
  # 春节
  date(2001, 1, 24), date(2001, 1, 25), date(2001, 1, 26), date(2001, 1, 27), date(2001, 1, 28), date(2001, 1, 29), date(2001, 1, 30),
  # 劳动节
  date(2001, 5, 1), date(2001, 5, 2), date(2001, 5, 3), date(2001, 5, 4), date(2001, 5, 5), date(2001, 5, 6), date(2001, 5, 7),
  # 国庆节
  date(2001, 10, 1), date(2001, 10, 2), date(2001, 10, 3), date(2001, 10, 4), date(2001, 10, 5), date(2001, 10, 6), date(2001, 10, 7)
])
# 2002年公众假日 (来源: https://zh.wikisource.org/wiki/%E5%9B%BD%E5%8A%A1%E9%99%A2%E5%8A%9E%E5%85%AC%E5%8E%85%E5%85%B3%E4%BA%8E2002%E5%B9%B4%E9%83%A8%E5%88%86%E8%8A%82%E5%81%87%E6%97%A5%E4%BC%91%E6%81%AF%E5%AE%89%E6%8E%92%E7%9A%84%E9%80%9A%E7%9F%A5)
holiday_date.extend([
  # 元旦
  date(2002, 1, 1), date(2002, 1, 2), date(2002, 1, 3),
  # 春节
  date(2002, 2, 12), date(2002, 2, 13), date(2002, 2, 14), date(2002, 2, 15), date(2002, 2, 16), date(2002, 2, 17), date(2002, 2, 18),
  # 劳动节
  date(2002, 5, 1), date(2002, 5, 2), date(2002, 5, 3), date(2002, 5, 4), date(2002, 5, 5), date(2002, 5, 6), date(2002, 5, 7),
  # 国庆节
  date(2002, 10, 1), date(2002, 10, 2), date(2002, 10, 3), date(2002, 10, 4), date(2002, 10, 5), date(2002, 10, 6), date(2002, 10, 7)
])
# 2003年公众假日 (来源: https://zh.wikisource.org/wiki/%E5%9B%BD%E5%8A%A1%E9%99%A2%E5%8A%9E%E5%85%AC%E5%8E%85%E5%85%B3%E4%BA%8E2003%E5%B9%B4%E9%83%A8%E5%88%86%E8%8A%82%E5%81%87%E6%97%A5%E4%BC%91%E6%81%AF%E5%AE%89%E6%8E%92%E7%9A%84%E9%80%9A%E7%9F%A5 及 https://zh.wikisource.org/wiki/%E5%9B%BD%E5%8A%A1%E9%99%A2%E5%8A%9E%E5%85%AC%E5%8E%85%E5%85%B3%E4%BA%8E2003%E5%B9%B4%E2%80%9C%E4%BA%94%E4%B8%80%E2%80%9D%E6%94%BE%E5%81%87%E8%B0%83%E4%BC%91%E5%AE%89%E6%8E%92%E7%9A%84%E9%80%9A%E7%9F%A5)
holiday_date.extend([
  # 元旦
  date(2003, 1, 1),
  # 春节
  date(2003, 2, 1), date(2003, 2, 2), date(2003, 2, 3), date(2003, 2, 4), date(2003, 2, 5), date(2003, 2, 6), date(2003, 2, 7),
  # 劳动节 (因“非典”疫情缩短)
  date(2003, 5, 1), date(2003, 5, 2), date(2003, 5, 3), date(2003, 5, 4), date(2003, 5, 5),
  # 国庆节
  date(2003, 10, 1), date(2003, 10, 2), date(2003, 10, 3), date(2003, 10, 4), date(2003, 10, 5), date(2003, 10, 6), date(2003, 10, 7)
])
# 2004年公众假日 (来源: https://zh.wikisource.org/wiki/%E5%9B%BD%E5%8A%A1%E9%99%A2%E5%8A%9E%E5%85%AC%E5%8E%85%E5%85%B3%E4%BA%8E2004%E5%B9%B4%E9%83%A8%E5%88%86%E8%8A%82%E5%81%87%E6%97%A5%E5%AE%89%E6%8E%92%E7%9A%84%E9%80%9A%E7%9F%A5)
holiday_date.extend([
  # 元旦
  date(2004, 1, 1),
  # 春节
  date(2004, 1, 22), date(2004, 1, 23), date(2004, 1, 24), date(2004, 1, 25), date(2004, 1, 26), date(2004, 1, 27), date(2004, 1, 28),
  # 劳动节
  date(2004, 5, 1), date(2004, 5, 2), date(2004, 5, 3), date(2004, 5, 4), date(2004, 5, 5), date(2004, 5, 6), date(2004, 5, 7),
  # 国庆节
  date(2004, 10, 1), date(2004, 10, 2), date(2004, 10, 3), date(2004, 10, 4), date(2004, 10, 5), date(2004, 10, 6), date(2004, 10, 7)
])
# 2005年公众假日 (来源: http://zqb.cyol.com/content/2004-12/21/content_1000540.htm)
holiday_date.extend([
  # 元旦
  date(2005, 1, 1), date(2005, 1, 2), date(2005, 1, 3),
  # 春节
  date(2005, 2, 9), date(2005, 2, 10), date(2005, 2, 11), date(2005, 2, 12), date(2005, 2, 13), date(2005, 2, 14), date(2005, 2, 15),
  # 劳动节
  date(2005, 5, 1), date(2005, 5, 2), date(2005, 5, 3), date(2005, 5, 4), date(2005, 5, 5), date(2005, 5, 6), date(2005, 5, 7),
  # 国庆节
  date(2005, 10, 1), date(2005, 10, 2), date(2005, 10, 3), date(2005, 10, 4), date(2005, 10, 5), date(2005, 10, 6), date(2005, 10, 7)
])
# 2006年公众假日 (来源: https://zh.wikisource.org/wiki/%E5%9B%BD%E5%8A%A1%E9%99%A2%E5%8A%9E%E5%85%AC%E5%8E%85%E5%85%B3%E4%BA%8E2006%E5%B9%B4%E9%83%A8%E5%88%86%E8%8A%82%E5%81%87%E6%97%A5%E5%AE%89%E6%8E%92%E7%9A%84%E9%80%9A%E7%9F%A5)
holiday_date.extend([
  # 元旦
  date(2006, 1, 1), date(2006, 1, 2), date(2006, 1, 3),
  # 春节
  date(2006, 1, 29), date(2006, 1, 30), date(2006, 1, 31), date(2006, 2, 1), date(2006, 2, 2), date(2006, 2, 3), date(2006, 2, 4),
  # 劳动节
  date(2006, 5, 1), date(2006, 5, 2), date(2006, 5, 3), date(2006, 5, 4), date(2006, 5, 5), date(2006, 5, 6), date(2006, 5, 7),
  # 国庆节
  date(2006, 10, 1), date(2006, 10, 2), date(2006, 10, 3), date(2006, 10, 4), date(2006, 10, 5), date(2006, 10, 6), date(2006, 10, 7)
])
# 2007年公众假日 (来源: https://www.gov.cn/gongbao/content/2007/content_503397.htm)
holiday_date.extend([
  # 元旦
  date(2007, 1, 1), date(2007, 1, 2), date(2007, 1, 3),
  # 春节
  date(2007, 2, 18), date(2007, 2, 19), date(2007, 2, 20), date(2007, 2, 21), date(2007, 2, 22), date(2007, 2, 23), date(2007, 2, 24),
  # 劳动节
  date(2007, 5, 1), date(2007, 5, 2), date(2007, 5, 3), date(2007, 5, 4), date(2007, 5, 5), date(2007, 5, 6), date(2007, 5, 7),
  # 国庆节
  date(2007, 10, 1), date(2007, 10, 2), date(2007, 10, 3), date(2007, 10, 4), date(2007, 10, 5), date(2007, 10, 6), date(2007, 10, 7)
])
# 2008年公众假日 (来源: https://www.gov.cn/gongbao/content/2008/content_859870.htm)
holiday_date.extend([
  # 元旦
  date(2007, 12, 30), date(2007, 12, 31), date(2008, 1, 1),
  # 春节
  date(2008, 2, 6), date(2008, 2, 7), date(2008, 2, 8), date(2008, 2, 9), date(2008, 2, 10), date(2008, 2, 11), date(2008, 2, 12),
  # 清明节
  date(2008, 4, 4), date(2008, 4, 5), date(2008, 4, 6),
  # 劳动节
  date(2008, 5, 1), date(2008, 5, 2), date(2008, 5, 3),
  # 端午节
  date(2008, 6, 7), date(2008, 6, 8), date(2008, 6, 9),
  # 中秋节
  date(2008, 9, 13), date(2008, 9, 14), date(2008, 9, 15),
  # 国庆节
  date(2008, 9, 29), date(2008, 9, 30), date(2008, 10, 1), date(2008, 10, 2), date(2008, 10, 3), date(2008, 10, 4), date(2008, 10, 5)
])
# 2009年公众假日 (来源: https://www.gov.cn/gongbao/content/2009/content_1277120.htm)
holiday_date.extend([
  # 元旦
  date(2009, 1, 1), date(2009, 1, 2), date(2009, 1, 3),
  # 春节
  date(2009, 1, 25), date(2009, 1, 26), date(2009, 1, 27), date(2009, 1, 28), date(2009, 1, 29), date(2009, 1, 30), date(2009, 1, 31),
  # 清明节
  date(2009, 4, 4), date(2009, 4, 5), date(2009, 4, 6),
  # 劳动节
  date(2009, 5, 1), date(2009, 5, 2), date(2009, 5, 3),
  # 端午节
  date(2009, 5, 28), date(2009, 5, 29), date(2009, 5, 30),
  # 国庆节、中秋节
  date(2009, 10, 1), date(2009, 10, 2), date(2009, 10, 3), date(2009, 10, 4), date(2009, 10, 5), date(2009, 10, 6), date(2009, 10, 7), date(2009, 10, 8)
])
# 2010年公众假日 (来源: https://www.gov.cn/zwgk/2009-12/08/content_1482691.htm)
holiday_date.extend([
  # 元旦
  date(2010, 1, 1), date(2010, 1, 2), date(2010, 1, 3),
  # 春节
  date(2010, 2, 13), date(2010, 2, 14), date(2010, 2, 15), date(2010, 2, 16), date(2010, 2, 17), date(2010, 2, 18), date(2010, 2, 19),
  # 清明节
  date(2010, 4, 3), date(2010, 4, 4), date(2010, 4, 5),
  # 劳动节
  date(2010, 5, 1), date(2010, 5, 2), date(2010, 5, 3),
  # 端午节
  date(2010, 6, 14), date(2010, 6, 15), date(2010, 6, 16),
  # 中秋节
  date(2010, 9, 22), date(2010, 9, 23), date(2010, 9, 24),
  # 国庆节
  date(2010, 10, 1), date(2010, 10, 2), date(2010, 10, 3), date(2010, 10, 4), date(2010, 10, 5), date(2010, 10, 6), date(2010, 10, 7)
])
# 2011年公众假日 (来源: https://www.gov.cn/gongbao/content/2010/content_1765282.htm)
holiday_date.extend([
  # 元旦
  date(2011, 1, 1), date(2011, 1, 2), date(2011, 1, 3),
  # 春节
  date(2011, 2, 2), date(2011, 2, 3), date(2011, 2, 4), date(2011, 2, 5), date(2011, 2, 6), date(2011, 2, 7), date(2011, 2, 8),
  # 清明节
  date(2011, 4, 3), date(2011, 4, 4), date(2011, 4, 5),
  # 劳动节
  date(2011, 4, 30), date(2011, 5, 1), date(2011, 5, 2),
  # 端午节
  date(2011, 6, 4), date(2011, 6, 5), date(2011, 6, 6),
  # 中秋节
  date(2011, 9, 10), date(2011, 9, 11), date(2011, 9, 12),
  # 国庆节
  date(2011, 10, 1), date(2011, 10, 2), date(2011, 10, 3), date(2011, 10, 4), date(2011, 10, 5), date(2011, 10, 6), date(2011, 10, 7)
])
# 2012年公众假日 (来源: https://www.gov.cn/gongbao/content/2011/content_2020918.htm)
holiday_date.extend([
  # 元旦
  date(2012, 1, 1), date(2012, 1, 2), date(2012, 1, 3),
  # 春节
  date(2012, 1, 22), date(2012, 1, 23), date(2012, 1, 24), date(2012, 1, 25), date(2012, 1, 26), date(2012, 1, 27), date(2012, 1, 28),
  # 清明节
  date(2012, 4, 2), date(2012, 4, 3), date(2012, 4, 4),
  # 劳动节
  date(2012, 4, 29), date(2012, 4, 30), date(2012, 5, 1),
  # 端午节
  date(2012, 6, 22), date(2012, 6, 23), date(2012, 6, 24),
  # 中秋节、国庆节
  date(2012, 9, 30), date(2012, 10, 1), date(2012, 10, 2), date(2012, 10, 3), date(2012, 10, 4), date(2012, 10, 5), date(2012, 10, 6), date(2012, 10, 7)
])
# 2013年公众假日 (来源: https://www.gov.cn/zwgk/2012-12/10/content_2286598.htm)
holiday_date.extend([
  # 元旦
  date(2013, 1, 1), date(2013, 1, 2), date(2013, 1, 3),
  # 春节
  date(2013, 2, 9), date(2013, 2, 10), date(2013, 2, 11), date(2013, 2, 12), date(2013, 2, 13), date(2013, 2, 14), date(2013, 2, 15),
  # 清明节
  date(2013, 4, 4), date(2013, 4, 5), date(2013, 4, 6),
  # 劳动节
  date(2013, 4, 29), date(2013, 4, 30), date(2013, 5, 1),
  # 端午节
  date(2013, 6, 10), date(2013, 6, 11), date(2013, 6, 12),
  # 中秋节
  date(2013, 9, 19), date(2013, 9, 20), date(2013, 9, 21),
  # 国庆节
  date(2013, 10, 1), date(2013, 10, 2), date(2013, 10, 3), date(2013, 10, 4), date(2013, 10, 5), date(2013, 10, 6), date(2013, 10, 7)
])
# 2014年公众假日 (来源: https://www.gov.cn/zwgk/2013-12/11/content_2546204.htm)
holiday_date.extend([
  # 元旦
  date(2014, 1, 1),
  # 春节
  date(2014, 1, 31), date(2014, 2, 1), date(2014, 2, 2), date(2014, 2, 3), date(2014, 2, 4), date(2014, 2, 5), date(2014, 2, 6),
  # 清明节
  date(2014, 4, 5), date(2014, 4, 6), date(2014, 4, 7),
  # 劳动节
  date(2014, 5, 1), date(2014, 5, 2), date(2014, 5, 3),
  # 端午节
  date(2014, 5, 31), date(2014, 6, 1), date(2014, 6, 2),
  # 中秋节
  date(2014, 9, 6), date(2014, 9, 7), date(2014, 9, 8),
  # 国庆节
  date(2014, 10, 1), date(2014, 10, 2), date(2014, 10, 3), date(2014, 10, 4), date(2014, 10, 5), date(2014, 10, 6), date(2014, 10, 7)
])
# 2015年公众假日 (来源: https://www.gov.cn/gongbao/content/2015/content_2799019.htm 及 https://www.gov.cn/zhengce/content/2015-05/13/content_9744.htm)
holiday_date.extend([
  # 元旦
  date(2015, 1, 1), date(2015, 1, 2), date(2015, 1, 3),
  # 春节
  date(2015, 2, 18), date(2015, 2, 19), date(2015, 2, 20), date(2015, 2, 21), date(2015, 2, 22), date(2015, 2, 23), date(2015, 2, 24),
  # 清明节
  date(2015, 4, 4), date(2015, 4, 5), date(2015, 4, 6),
  # 劳动节
  date(2015, 5, 1), date(2015, 5, 2), date(2015, 5, 3),
  # 端午节
  date(2015, 6, 20), date(2015, 6, 21), date(2015, 6, 22),
  # 抗战胜利70周年纪念日
  date(2015, 9, 3),
  # 中秋节
  date(2015, 9, 27),
  # 国庆节
  date(2015, 10, 1), date(2015, 10, 2), date(2015, 10, 3), date(2015, 10, 4), date(2015, 10, 5), date(2015, 10, 6), date(2015, 10, 7)
])
# 2016年公众假日 (来源: http://politics.people.com.cn/n/2015/1211/c1001-27913395.html)
holiday_date.extend([
  # 元旦
  date(2016, 1, 1), date(2016, 1, 2), date(2016, 1, 3),
  # 春节
  date(2016, 2, 7), date(2016, 2, 8), date(2016, 2, 9), date(2016, 2, 10), date(2016, 2, 11), date(2016, 2, 12), date(2016, 2, 13),
  # 清明节
  date(2016, 4, 2), date(2016, 4, 3), date(2016, 4, 4),
  # 劳动节
  date(2016, 4, 30), date(2016, 5, 1), date(2016, 5, 2),
  # 端午节
  date(2016, 6, 9), date(2016, 6, 10), date(2016, 6, 11),
  # 中秋节
  date(2016, 9, 15), date(2016, 9, 16), date(2016, 9, 17),
  # 国庆节
  date(2016, 10, 1), date(2016, 10, 2), date(2016, 10, 3), date(2016, 10, 4), date(2016, 10, 5), date(2016, 10, 6), date(2016, 10, 7)
])
# 2017年公众假日 (来源: http://www.xinhuanet.com/politics/2016-12/01/c_129386943.htm)
holiday_date.extend([
  # 元旦
  date(2016, 12, 31), date(2017, 1, 1), date(2017, 1, 2),
  # 春节
  date(2017, 1, 27), date(2017, 1, 28), date(2017, 1, 29), date(2017, 1, 30), date(2017, 1, 31), date(2017, 2, 1), date(2017, 2, 2),
  # 清明节
  date(2017, 4, 2), date(2017, 4, 3), date(2017, 4, 4),
  # 劳动节
  date(2017, 4, 29), date(2017, 4, 30), date(2017, 5, 1),
  # 端午节
  date(2017, 5, 28), date(2017, 5, 29), date(2017, 5, 30),
  # 中秋节、国庆节
  date(2017, 10, 1), date(2017, 10, 2), date(2017, 10, 3), date(2017, 10, 4), date(2017, 10, 5), date(2017, 10, 6), date(2017, 10, 7), date(2017, 10, 8)
])
# 2018年公众假日 (来源: https://www.gov.cn/zhengce/content/2017-11/30/content_5243579.htm)
holiday_date.extend([
  # 元旦
  date(2017, 12, 30), date(2017, 12, 31), date(2018, 1, 1),
  # 春节
  date(2018, 2, 15), date(2018, 2, 16), date(2018, 2, 17), date(2018, 2, 18), date(2018, 2, 19), date(2018, 2, 20), date(2018, 2, 21),
  # 清明节
  date(2018, 4, 5), date(2018, 4, 6), date(2018, 4, 7),
  # 劳动节
  date(2018, 4, 29), date(2018, 4, 30), date(2018, 5, 1),
  # 端午节
  date(2018, 6, 16), date(2018, 6, 17), date(2018, 6, 18),
  # 中秋节
  date(2018, 9, 22), date(2018, 9, 23), date(2018, 9, 24),
  # 国庆节
  date(2018, 10, 1), date(2018, 10, 2), date(2018, 10, 3), date(2018, 10, 4), date(2018, 10, 5), date(2018, 10, 6), date(2018, 10, 7)
])
# 2019年公众假日 (来源: https://www.gov.cn/zhengce/content/2018-12/06/content_5346276.htm)
holiday_date.extend([
  # 元旦
  date(2018, 12, 30), date(2018, 12, 31), date(2019, 1, 1),
  # 春节
  date(2019, 2, 4), date(2019, 2, 5), date(2019, 2, 6), date(2019, 2, 7), date(2019, 2, 8), date(2019, 2, 9), date(2019, 2, 10),
  # 清明节
  date(2019, 4, 5), date(2019, 4, 6), date(2019, 4, 7),
  # 劳动节 (当年3月临时调整为4天，此处为原始通知)
  date(2019, 5, 1),
  # 端午节
  date(2019, 6, 7), date(2019, 6, 8), date(2019, 6, 9),
  # 中秋节
  date(2019, 9, 13), date(2019, 9, 14), date(2019, 9, 15),
  # 国庆节
  date(2019, 10, 1), date(2019, 10, 2), date(2019, 10, 3), date(2019, 10, 4), date(2019, 10, 5), date(2019, 10, 6), date(2019, 10, 7)
])
# 2020年公众假日 (来源: https://www.gov.cn/zhengce/content/2019-11/21/content_5454164.htm)
holiday_date.extend([
  # 元旦
  date(2020, 1, 1),
  # 春节 (因新冠疫情延长至2月2日)
  date(2020, 1, 24), date(2020, 1, 25), date(2020, 1, 26), date(2020, 1, 27), date(2020, 1, 28), date(2020, 1, 29), date(2020, 1, 30), date(2020, 1, 31), date(2020, 2, 1),
  date(2020, 2, 2),
  # 清明节
  date(2020, 4, 4), date(2020, 4, 5), date(2020, 4, 6),
  # 劳动节
  date(2020, 5, 1), date(2020, 5, 2), date(2020, 5, 3), date(2020, 5, 4), date(2020, 5, 5),
  # 端午节
  date(2020, 6, 25), date(2020, 6, 26), date(2020, 6, 27),
  # 国庆节、中秋节
  date(2020, 10, 1), date(2020, 10, 2), date(2020, 10, 3), date(2020, 10, 4), date(2020, 10, 5), date(2020, 10, 6), date(2020, 10, 7), date(2020, 10, 8)
])
# 2021年公众假日 (来源: https://www.gov.cn/zhengce/content/2020-11/25/content_5564127.htm)
holiday_date.extend([
  # 元旦
  date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3),
  # 春节
  date(2021, 2, 11), date(2021, 2, 12), date(2021, 2, 13), date(2021, 2, 14), date(2021, 2, 15), date(2021, 2, 16), date(2021, 2, 17),
  # 清明节
  date(2021, 4, 3), date(2021, 4, 4), date(2021, 4, 5),
  # 劳动节
  date(2021, 5, 1), date(2021, 5, 2), date(2021, 5, 3), date(2021, 5, 4), date(2021, 5, 5),
  # 端午节
  date(2021, 6, 12), date(2021, 6, 13), date(2021, 6, 14),
  # 中秋节
  date(2021, 9, 19), date(2021, 9, 20), date(2021, 9, 21),
  # 国庆节
  date(2021, 10, 1), date(2021, 10, 2), date(2021, 10, 3), date(2021, 10, 4), date(2021, 10, 5), date(2021, 10, 6), date(2021, 10, 7)
])
# 2022年公众假日 (来源: https://www.gov.cn/zhengce/content/2021-10/25/content_5644835.htm)
holiday_date.extend([
  # 元旦
  date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3),
  # 春节
  date(2022, 1, 31), date(2022, 2, 1), date(2022, 2, 2), date(2022, 2, 3), date(2022, 2, 4), date(2022, 2, 5), date(2022, 2, 6),
  # 清明节
  date(2022, 4, 3), date(2022, 4, 4), date(2022, 4, 5),
  # 劳动节
  date(2022, 4, 30), date(2022, 5, 1), date(2022, 5, 2), date(2022, 5, 3), date(2022, 5, 4),
  # 端午节
  date(2022, 6, 3), date(2022, 6, 4), date(2022, 6, 5),
  # 中秋节
  date(2022, 9, 10), date(2022, 9, 11), date(2022, 9, 12),
  # 国庆节
  date(2022, 10, 1), date(2022, 10, 2), date(2022, 10, 3), date(2022, 10, 4), date(2022, 10, 5), date(2022, 10, 6), date(2022, 10, 7)
])
# 2023年公众假日 (来源: https://www.gov.cn/zhengce/content/2022-12/08/content_5730844.htm)
holiday_date.extend([
  # 元旦
  date(2022, 12, 31), date(2023, 1, 1), date(2023, 1, 2),
  # 春节
  date(2023, 1, 21), date(2023, 1, 22), date(2023, 1, 23), date(2023, 1, 24), date(2023, 1, 25), date(2023, 1, 26), date(2023, 1, 27),
  # 清明节
  date(2023, 4, 5),
  # 劳动节
  date(2023, 4, 29), date(2023, 4, 30), date(2023, 5, 1), date(2023, 5, 2), date(2023, 5, 3),
  # 端午节
  date(2023, 6, 22), date(2023, 6, 23), date(2023, 6, 24),
  # 中秋节、国庆节
  date(2023, 9, 29), date(2023, 9, 30), date(2023, 10, 1), date(2023, 10, 2), date(2023, 10, 3), date(2023, 10, 4), date(2023, 10, 5), date(2023, 10, 6)
])
# 2024年公众假日 (来源: https://www.gov.cn/zhengce/content/202310/content_6911527.htm)
holiday_date.extend([
  # 元旦
  date(2023, 12, 30), date(2023, 12, 31), date(2024, 1, 1),
  # 春节
  date(2024, 2, 10), date(2024, 2, 11), date(2024, 2, 12), date(2024, 2, 13), date(2024, 2, 14), date(2024, 2, 15), date(2024, 2, 16), date(2024, 2, 17),
  # 清明节
  date(2024, 4, 4), date(2024, 4, 5), date(2024, 4, 6),
  # 劳动节
  date(2024, 5, 1), date(2024, 5, 2), date(2024, 5, 3), date(2024, 5, 4), date(2024, 5, 5),
  # 端午节
  date(2024, 6, 8), date(2024, 6, 9), date(2024, 6, 10),
  # 中秋节
  date(2024, 9, 15), date(2024, 9, 16), date(2024, 9, 17),
  # 国庆节
  date(2024, 10, 1), date(2024, 10, 2), date(2024, 10, 3), date(2024, 10, 4), date(2024, 10, 5), date(2024, 10, 6), date(2024, 10, 7)
])
# 2025年公众假日 (来源: https://www.gov.cn/zhengce/content/202411/content_6986382.htm)
holiday_date.extend([
  # 元旦
  date(2025, 1, 1),
  # 春节
  date(2025, 1, 28), date(2025, 1, 29), date(2025, 1, 30), date(2025, 1, 31), date(2025, 2, 1), date(2025, 2, 2), date(2025, 2, 3), date(2025, 2, 4),
  # 清明节
  date(2025, 4, 4), date(2025, 4, 5), date(2025, 4, 6),
  # 劳动节
  date(2025, 5, 1), date(2025, 5, 2), date(2025, 5, 3), date(2025, 5, 4), date(2025, 5, 5),
  # 端午节
  date(2025, 5, 31), date(2025, 6, 1), date(2025, 6, 2),
  # 国庆节、中秋节
  date(2025, 10, 1), date(2025, 10, 2), date(2025, 10, 3), date(2025, 10, 4), date(2025, 10, 5), date(2025, 10, 6), date(2025, 10, 7), date(2025, 10, 8)
])
# 2026年公众假日 (来源: https://www.gov.cn/zhengce/content/202511/content_7047090.htm)
holiday_date.extend([
  # 元旦
  date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3),
  # 春节
  date(2026, 2, 15), date(2026, 2, 16), date(2026, 2, 17), date(2026, 2, 18), date(2026, 2, 19), date(2026, 2, 20), date(2026, 2, 21), date(2026, 2, 22), date(2026, 2, 23),
  # 清明节
  date(2026, 4, 4), date(2026, 4, 5), date(2026, 4, 6),
  # 劳动节
  date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 3), date(2026, 5, 4), date(2026, 5, 5),
  # 端午节
  date(2026, 6, 19), date(2026, 6, 20), date(2026, 6, 21),
  # 中秋节
  date(2026, 9, 25), date(2026, 9, 26), date(2026, 9, 27),
  # 国庆节
  date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 3), date(2026, 10, 4), date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7)
])

# 转换为set以提高查找性能 - O(1) vs O(n)
holiday_date_set = frozenset(holiday_date)

@lru_cache(maxsize=1024)
def is_trading_day(target_date: date = None) -> bool:
  """
  判断是否为交易日
  优化版本：使用LRU缓存和set查找提升性能
  """
  if target_date is None:
    target_date = date.today()

  # 快速检查：周末直接返回False
  if target_date.weekday() >= 5:  # 5=周六, 6=周日
    return False

  # 使用set查找，O(1)时间复杂度
  return target_date not in holiday_date_set
