import tkinter as tk
from tkinter import ttk
import threading
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re

import requests
from bs4 import BeautifulSoup


class Data_Crawler():
    def __init__(self):
        self.total_turnover = 0.0
        self.turnover_data = ""
        self.sector_data = []
        self.fund_data = []
        self.sign_data = []
        self.link_list = ["https://q.10jqka.com.cn/zs/detail/code/1A0001/",
                          "https://q.10jqka.com.cn/zs/detail/code/399001/",
                          "https://data.10jqka.com.cn/funds/gnzjl/#refCountId=data_55f13c2c_254",
                          "https://www.iwencai.com/unifiedwap/result?w=%E5%AF%B9%E5%BD%93%E5%A4%A9%E7%9A%84%E6%A6%82%E5%BF%B5%E6%9D%BF%E5%9D%97%E6%8C%89%E6%B6%A8%E5%B9%85%E8%BF%9B%E8%A1%8C%E6%8E%92%E5%BA%8F%20%E4%B8%A4%E4%B8%AA%E4%BA%A4%E6%98%93%E6%97%A5%20%E6%B6%A8%E5%81%9C%E6%95%B0%20%E4%B8%A4%E4%B8%AA%E4%BA%A4%E6%98%93%E6%97%A5%20%E6%88%90%E5%88%86%E8%82%A1%E6%95%B0%20%E5%BD%93%E5%A4%A9%E6%88%90%E4%BA%A4%E9%A2%9D%E3%80%82%E6%88%90%E5%88%86%E8%82%A1%E6%95%B0%E9%87%8F%E5%A4%A7%E4%BA%8E80%E7%9A%84%E6%8E%92%E9%99%A4&querytype=zhishu",
                          "https://www.iwencai.com/unifiedwap/result?w=000%E5%92%8C600%E5%BC%80%E5%A4%B4%20%E4%B8%A4%E4%B8%AA%E4%BA%A4%E6%98%93%E6%97%A5%20%E6%B6%A8%E5%B9%85%20%E4%B8%A4%E4%B8%AA%E4%BA%A4%E6%98%93%E6%97%A5%20%E5%A4%A7%E5%8D%95%E5%87%80%E9%87%8F%20%E4%B8%A4%E4%B8%AA%E4%BA%A4%E6%98%93%E6%97%A5%20%E5%A4%A7%E5%8D%95%E5%87%80%E9%A2%9D%E3%80%81%E6%88%90%E4%BA%A4%E9%A2%9D%20%E6%8C%89%E5%A4%A7%E5%8D%95%E5%87%80%E9%87%8F%E7%8E%AF%E6%AF%94%E6%8E%92%E5%BA%8F&querytype=stock",
                          "https://www.iwencai.com/unifiedwap/result?w=600%E5%92%8C000%E5%BC%80%E5%A4%B4%20%E5%87%80%E9%87%8F%E4%B8%BA%E8%B4%9F%20%E6%B6%A8%E8%B7%8C%E5%B9%85%E5%A4%A7%E4%BA%8E0%20%E6%8C%89%E5%87%80%E9%87%8F%E9%80%86%E5%BA%8F%20%E4%B8%8A%E4%B8%AA%E4%BA%A4%E6%98%93%E6%97%A5%E6%B6%A8%E8%B7%8C%E5%B9%85%E5%A4%A7%E4%BA%8E0%20%E5%87%80%E9%87%8F%E7%8E%AF%E6%AF%94%E5%A4%A7%E4%BA%8E0%20dma%E4%B9%B0%E5%85%A5%E4%BF%A1%E5%8F%B7&querytype=stock",
                          "https://www.iwencai.com/unifiedwap/result?w=600%E5%92%8C000%E5%BC%80%E5%A4%B4%20%E5%87%80%E9%87%8F%E4%B8%BA%E8%B4%9F%20%E6%B6%A8%E8%B7%8C%E5%B9%85%E5%A4%A7%E4%BA%8E0%20%E6%8C%89%E5%87%80%E9%87%8F%E9%80%86%E5%BA%8F%20%E4%B8%8A%E4%B8%AA%E4%BA%A4%E6%98%93%E6%97%A5%E6%B6%A8%E8%B7%8C%E5%B9%85%E5%A4%A7%E4%BA%8E0%20%E5%87%80%E9%87%8F%E7%8E%AF%E6%AF%94%E5%A4%A7%E4%BA%8E0%20%E8%BF%915%E4%B8%AA%E4%BA%A4%E6%98%93%E6%97%A5%E6%8C%AF%E5%B9%85%E9%AB%98%E4%BA%8E10%E4%B8%94%E4%BD%8E%E4%BA%8E20%20%E8%BF%913%E4%B8%AA%E4%BA%A4%E6%98%93%E6%97%A5%E6%B6%A8%E5%B9%85%E9%AB%98%E4%BA%8E6&querytype=stock",
                          "https://www.iwencai.com/unifiedwap/result?w=600%E5%92%8C000%E5%BC%80%E5%A4%B4%20%E6%B6%A8%E8%B7%8C%E5%B9%85%E5%A4%A7%E4%BA%8E0%20%E6%8C%89%E5%87%80%E9%87%8F%E9%80%86%E5%BA%8F%20%E4%B8%8A%E4%B8%AA%E4%BA%A4%E6%98%93%E6%97%A5%E6%B6%A8%E8%B7%8C%E5%B9%85%E5%A4%A7%E4%BA%8E0%20%E5%87%80%E9%87%8F%E7%8E%AF%E6%AF%94%E5%A4%A7%E4%BA%8E0%20%E4%B8%BB%E5%8A%9B%E4%B9%B0%E5%85%A5%E5%A4%A7%E4%BA%8E2000%E4%B8%87%20%E5%87%80%E9%A2%9D%E3%80%81%E4%BA%A4%E6%98%93%E9%A2%9D&querytype=stock"
                          ]
        self.xpath_list = ['html > body > div:nth-of-type(2) > div:nth-of-type(3) > div:nth-of-type(2) > div > div > div:nth-of-type(1) > div:nth-of-type(2) > dl:nth-of-type(7) > dd',
                           ]
        self.headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0"
        }

        options = webdriver.ChromeOptions()
        options.add_argument("--headless")  # 无头模式
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0")

        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self.driver.set_window_size(3000, 2000)

    def fetch_data(self, url, selector, use_CCS=True):
        # 发送HTTP请求
        response = requests.get(url, headers=self.headers)
        # 检查请求是否成功
        if response.status_code == 200:
            # 解析HTML内容
            soup = BeautifulSoup(response.text, 'html.parser')
            if use_CCS:
            # 使用CSS选择器定位元素
                return soup.select_one(selector)
            else:
                return soup

    def fetch_dynamic_data(self, url, by, name):
        self.driver.get(url)
        WebDriverWait(self.driver, 5).until(EC.presence_of_element_located((by, name)))
        content = self.driver.find_elements(by, name)
        texts = [element.text for element in content]
        return texts

    def fetch_data_name(self):
        content = {}
        k=5
        for i in range(1, 50):
            xpath = '//*[@id="iwc-table-container"]/div[5]/div[1]/div[1]/ul/li[{}]'.format(i)
            try:
                element = self.driver.find_element(By.XPATH, xpath)
                ls = element.text.split('\n')
                content.setdefault(ls[0],k)
                k+=len(ls) > 2
                content.setdefault(ls[0]+'2', k)
                k+=1
            except:
                break
        return content

    def get_turnover(self):
        shen_turnover = eval(self.fetch_data(self.link_list[0],self.xpath_list[0]).text)
        hu_turnover = eval(self.fetch_data(self.link_list[1],self.xpath_list[0]).text)
        self.total_turnover = round(shen_turnover + hu_turnover,2)
        return f"  深：{shen_turnover}  |  沪：{hu_turnover}  |  总：{self.total_turnover}"

    def get_fund(self,):
        fund_data = self.fetch_dynamic_data(self.link_list[4], By.TAG_NAME, "td")
        name = self.fetch_data_name()

        cycle = max(name.values())
        if len(fund_data) < cycle + 2:
            return []
        fund_list = []
        idx = [3, 4, name['dde大单净量(%)'], name['dde大单净量(%)2'], name['成交额(元)'], name['dde大单净额(元)'],
               name['dde大单净额(元)2'], name['涨跌幅:前复权(%)'], name['涨跌幅:前复权(%)2']]
        for i in range(50):
            tmp = []
            for j in idx:
                if fund_data[cycle * i + j][-1] == "万" and j != 4:
                    fund_data[cycle * i + j] = str(round(eval(fund_data[cycle * i + j][:-1].replace(",", ""))/10000,2)) + "亿"
                tmp.append(fund_data[cycle * i + j])
            data = (tmp[0]+'-'+tmp[1], tmp[7]+' ← '+tmp[8], tmp[5][:-1]+' ← '+tmp[6][:-1], tmp[2]+' ← '+tmp[3],f"{eval(tmp[2]+'/'+tmp[7]+'1')*10:.2f} ← {eval(tmp[3]+'/'+tmp[8]+'1')*10:.2f}",f"{tmp[4]} {eval(tmp[4][:-1])/self.total_turnover*1000:.2f}‰")
            fund_list.append(data)
            if fund_data[3] == fund_data[cycle*(i+1)+3]:
                break
        return fund_list

    def get_sector(self):
        sector_data = self.fetch_dynamic_data(self.link_list[3],By.TAG_NAME, "td")
        name = self.fetch_data_name()
        cycle = max(name.values())
        if len(sector_data) < cycle + 2:
            return []
        idx = [4,name['涨停家数(家)'],name['涨停家数(家)2'],name['成分股总数(家)'],name['成分股总数(家)2'],name['成交额(元)']]
        sector_list = []
        for i in range(20):
            tmp = []
            for j in idx:
                tmp.append(sector_data[cycle*i+j])
            data = (tmp[0],f"{tmp[1]}/{tmp[3]} {int(eval(tmp[1]+'/'+tmp[3])*100)}%",
                    f"{tmp[2]}/{tmp[4]} {int(eval(tmp[2]+'/'+tmp[4])*100)}%",
                    f"{tmp[5][:-1]} {eval(tmp[5][:-1].replace(',',''))/self.total_turnover*100:.2f}%",
                    f"{eval(tmp[5][:-1].replace(',', '')+'/'+tmp[3]):.2f}")
            sector_list.append(data)
        return sector_list

    def get_sign(self,link,last):
        sign_data = self.fetch_dynamic_data(self.link_list[link],By.TAG_NAME, "td")
        name = self.fetch_data_name()
        cycle = max(name.values())
        sign_list = []
        if len(sign_data) < cycle + 2:
            return sign_list
        idx = [3,4,name['涨跌幅:前复权(%)'],name['dde大单净量(%)'],name[last]]
        for i in range(50):
            tmp = []
            for j in idx:
                tmp.append(sign_data[cycle*i+j])
            data = (tmp[0]+'-'+tmp[1], tmp[2],tmp[3],f"{eval(tmp[3]+'/'+tmp[2]+'1')*10:.2f}",last[:4]+':'+tmp[4])
            sign_list.append(data)
            if sign_data[3] == sign_data[cycle*(i+1)+3]:
                break
        return sign_list


    def refresh_data(self):
        self.turnover_data = self.get_turnover()
        self.sector_data = self.get_sector()
        self.fund_data = self.get_fund()
        self.sign_data = (self.get_sign(5, '技术形态') +
                          self.get_sign(6, '区间涨跌幅:前复权(%)') +
                          self.get_sign(7, '主力资金流向(元)'))




class FinanceTracker(Data_Crawler):
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.sort_col = -1
        self.sort_reverse = True

        # 界面设计
        master.title("金融跟踪系统")
        master.geometry('1400x850')
        self.refresh_label = tk.Label(master, text="休市中", font=("Arial", 12))
        self.refresh_label.grid(row=0, column=0)

        # 设置主容器
        self.main_frame = ttk.Frame(master)
        self.main_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)  # 改为 grid()

        # 左侧板块跟踪
        self.left_frame = ttk.LabelFrame(self.main_frame, text=" 板块跟踪 ", padding=10)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # 右侧资金跟踪
        self.right_frame = ttk.LabelFrame(self.main_frame, text=" 资金跟踪 ", padding=10)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # 配置网格布局权重
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        # 刷新模块
        self.refresh_data()
        self.init_left_top_panel()
        self.init_left_down_panel()
        self.init_right_panel()
        self.auto_refresh_thread = threading.Thread(target=self.auto_refresh, daemon=True)
        self.auto_refresh_thread.start()

    def init_left_top_panel(self):
        """初始化左侧板块跟踪面板"""
        # 创建树状表格
        columns = ("板块名称", "涨停/总数", "昨日", "成交额", "平均成交")
        self.stock_tree = ttk.Treeview(
            self.left_frame,
            columns=columns,
            show="headings",
            height=12
        )

        # 设置列标题
        for col in columns:
            self.stock_tree.heading(col, text=col, command=lambda _col=col: self.sort_by_column(self.stock_tree,_col))
            self.stock_tree.column(col, width=110, anchor=tk.CENTER)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(self.left_frame, orient=tk.VERTICAL, command=self.stock_tree.yview)
        self.stock_tree.configure(yscrollcommand=scrollbar.set)

        # 布局
        self.stock_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # 添加示例数据
        sample_data = self.sector_data
        for item in sample_data:
            self.stock_tree.insert("", tk.END, values=item)

        # 配置布局权重
        self.left_frame.columnconfigure(0, weight=1)
        self.left_frame.rowconfigure(0, weight=1)

    def init_left_down_panel(self):
        """初始化左侧板块跟踪面板"""
        # 创建树状表格
        columns = ("股票代码", "涨幅", "净量","压力系数", "分析")
        self.sign_tree = ttk.Treeview(
            self.left_frame,
            columns=columns,
            show="headings",
            height=23
        )

        # 设置列标题
        for col in columns:
            self.sign_tree.heading(col, text=col, command=lambda _col=col: self.sort_by_column(self.sign_tree,_col))
            self.sign_tree.column(col, width=110, anchor=tk.CENTER)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(self.left_frame, orient=tk.VERTICAL, command=self.sign_tree.yview)
        self.sign_tree.configure(yscrollcommand=scrollbar.set)

        # 布局
        self.sign_tree.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")

        # 添加示例数据
        sample_data = self.sign_data
        for item in sample_data:
            self.sign_tree.insert("", tk.END, values=item)

        # 配置布局权重
        self.left_frame.columnconfigure(0, weight=1)
        self.left_frame.rowconfigure(1, weight=1)

    def init_right_panel(self):
        """初始化右侧资金跟踪面板"""
        # 使用网格布局
        self.right_frame.grid_columnconfigure(0, weight=1)

        # 资金概要
        summary_frame = ttk.Frame(self.right_frame)
        summary_frame.grid(row=0, column=0, sticky="ew", pady=5)

        ttk.Label(summary_frame, text="总成交额:").grid(row=1, column=0, sticky="w")
        ttk.Label(summary_frame, text="{}".format(self.turnover_data), foreground="green").grid(row=1, column=1, sticky="e")

        # 账户列表
        account_frame = ttk.LabelFrame(self.right_frame, text="资金明细")
        account_frame.grid(row=1, column=0, sticky="nsew", pady=5)

        # 创建树状表格
        columns = ("股票代码", "涨幅", "净额", "净量", "压力系数", "成交额")
        self.account_tree = ttk.Treeview(
            account_frame,
            columns=columns,
            show="headings",
            height=30
        )

        for col in columns:
            self.account_tree.heading(col, text=col, command=lambda _col=col: self.sort_by_column(self.account_tree,_col))
            self.account_tree.column(col, width=120, anchor=tk.CENTER)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(account_frame, orient=tk.VERTICAL, command=self.account_tree.yview)
        self.account_tree.configure(yscrollcommand=scrollbar.set)

        # 布局
        self.account_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # 添加资金数据
        fund_data = self.fund_data
        for item in fund_data:
            self.account_tree.insert("", tk.END, values=item)

        # 配置布局权重
        account_frame.columnconfigure(0, weight=1)
        account_frame.rowconfigure(0, weight=1)
        self.right_frame.rowconfigure(1, weight=1)


    def sort_by_column(self, tree, col):
        """点击列标题时排序"""
        # 获取当前所有行数据
        rows = [tree.item(item)["values"] for item in tree.get_children()]
        col_index = tree["columns"].index(col)

        if col in {"平均成交"}:
            method = lambda x:float(x[col_index])
        elif col in {"涨停/总数", "昨日", "成交额"}:
            method = lambda x: eval(re.findall(r"\s([^%‰]+)[%‰]", x[col_index])[0])
        elif col in {"涨幅", "净额", "压力系数"}:
            method = lambda x: float(x[col_index].split(' ')[0])
        else:
            method = lambda x: x[col_index]

        # 判断当前是否为升序或降序
        if self.sort_col == col_index:
            self.sort_reverse = not self.sort_reverse  # 反转排序顺序
        else:
            self.sort_col = col_index
            self.sort_reverse = True

        # 按照列进行排序
        sorted_rows = sorted(rows, key=method, reverse=self.sort_reverse)

        # 清空当前树状表格并重新插入排序后的数据
        for item in tree.get_children():
            tree.delete(item)

        for row in sorted_rows:
            tree.insert("", tk.END, values=row)


    def auto_refresh(self):
        while True:
            if 9 < datetime.now().hour < 15:
                self.refresh_thread = threading.Thread(target = self.refresh_data,daemon=True)
                self.refresh_thread.start()
                self.refresh_thread.join()
                self.init_left_top_panel()
                self.init_left_down_panel()
                self.init_right_panel()
                current = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.refresh_label.config(text=f"上次更新时间: {current}")
                time.sleep(100)
            else:
                self.refresh_label.config(text=f"休市中")
                time.sleep(1800)




def main():
    root = tk.Tk()
    FinanceTracker(root)
    root.mainloop()

if __name__ == "__main__":
    main()