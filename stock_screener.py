"""
高胜率选股策略 - 按 README.md 策略执行
策略一：龙头共振 (板块+个股双重共振)
策略二：启动信号 (横盘突破第一天买入)
依赖: pip install akshare pandas numpy
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import time
import argparse

warnings.filterwarnings("ignore")

# ========== 第一层：市场环境过滤 ==========

def check_market_environment(
    min_sh_pct=0.0,
    require_up_majority=True,
    min_north_net=0.0,
):
    """
    市场环境过滤 (参数可调):
    :param min_sh_pct: 沪指最低涨幅要求, 默认 0.0 (即 > 0%)
    :param require_up_majority: 是否要求上涨家数 > 下跌家数, 默认 True
    :param min_north_net: 北向资金最低净流入(元), 默认 0 (即 > 0)
    """
    print("\n" + "=" * 60)
    print("  第一层：市场环境过滤")
    print(f"  参数: 沪指>{min_sh_pct}% | 涨多跌少:{require_up_majority} | 北向>{min_north_net/1e8:.1f}亿")
    print("=" * 60)

    # 沪指涨幅
    try:
        sh_df = ak.stock_zh_index_spot_em()
        sh_row = sh_df[sh_df["代码"] == "000001"]
        sh_pct = float(sh_row["涨跌幅"].values[0])
        passed = sh_pct > min_sh_pct
        print(f"  沪指涨幅: {sh_pct:.2f}% (要求>{min_sh_pct}%)  {'✅ 通过' if passed else '❌ 不满足'}")
    except Exception as e:
        print(f"  ⚠️ 获取沪指失败({e}), 假设通过")
        sh_pct = min_sh_pct + 0.1
        passed = True

    if not passed:
        print("  ❌ 大盘不达标，策略建议空仓观望")
        return False, {}

    # 上涨/下跌家数
    try:
        spot_df = ak.stock_zh_a_spot_em()
        spot_df["涨跌幅"] = pd.to_numeric(spot_df["涨跌幅"], errors="coerce")
        up_count = len(spot_df[spot_df["涨跌幅"] > 0])
        down_count = len(spot_df[spot_df["涨跌幅"] < 0])
        print(f"  上涨家数: {up_count}, 下跌家数: {down_count}  {'✅' if up_count > down_count else '❌'}")
    except Exception as e:
        print(f"  ⚠️ 获取涨跌家数失败({e})")
        up_count, down_count = 1, 0
        spot_df = None

    if require_up_majority and up_count <= down_count:
        print("  ❌ 下跌家数多于上涨，市场偏弱")
        return False, {}

    # 北向资金
    try:
        north_df = ak.stock_hsgt_north_net_flow_in_em(symbol="北向")
        if north_df is not None and len(north_df) > 0:
            north_net = float(north_df.iloc[-1]["当日净流入"])
            passed_north = north_net > min_north_net
            print(f"  北向资金净流入: {north_net/1e8:.2f}亿 (要求>{min_north_net/1e8:.1f}亿)  {'✅' if passed_north else '❌'}")
            if not passed_north:
                print("  ⚠️ 北向资金不达标，降低仓位信号（不阻断）")
        else:
            north_net = min_north_net + 1
    except Exception as e:
        print(f"  ⚠️ 北向数据获取失败({e}), 跳过")
        north_net = min_north_net + 1

    print("  ✅ 市场环境通过！")
    return True, {"spot_df": spot_df, "sh_pct": sh_pct}


# ========== 第二层：板块共振 ==========

def find_hot_sectors():
    """
    板块共振:
    - 板块当日涨幅 > 1.5%
    - 板块内涨幅 > 5% 的个股 >= 3 只
    """
    print("\n" + "=" * 60)
    print("  第二层：板块共振筛选")
    print("=" * 60)

    try:
        sector_df = ak.stock_board_industry_name_em()
        sector_df["涨跌幅"] = pd.to_numeric(sector_df["涨跌幅"], errors="coerce")
        hot_sectors = sector_df[sector_df["涨跌幅"] > 1.5]
        print(f"  涨幅 > 1.5% 的板块: {len(hot_sectors)} 个")
    except Exception as e:
        print(f"  ⚠️ 板块数据获取失败({e})")
        return []

    if hot_sectors.empty:
        print("  ❌ 无热点板块, 策略不触发")
        return []

    valid_sectors = []
    for _, row in hot_sectors.iterrows():
        board_name = row["板块名称"]
        board_pct = row["涨跌幅"]
        try:
            cons = ak.stock_board_industry_cons_em(symbol=board_name)
            if cons is None or cons.empty:
                continue
            cons["涨跌幅"] = pd.to_numeric(cons["涨跌幅"], errors="coerce")
            strong_count = len(cons[cons["涨跌幅"] > 5])
            if strong_count >= 3:
                valid_sectors.append({
                    "name": board_name,
                    "pct": board_pct,
                    "strong_count": strong_count,
                    "stocks": cons,
                })
                print(f"  ✓ {board_name} 涨{board_pct:.2f}%, 强势股{strong_count}只")
            time.sleep(0.1)
        except Exception:
            continue

    print(f"  ✅ 共振板块: {len(valid_sectors)} 个")
    return valid_sectors


# ========== 第三层：个股精选 (策略一) ==========

def strategy1_stock_filter(hot_sectors, spot_df):
    """
    龙头共振 - 个股精选:
    - 近60日高点附近
    - 成交量 > 20日均量 × 1.5
    - 今日涨幅 3%~8%
    - 均线多头: MA5 > MA10 > MA20 > MA60
    - 非ST/非科创/非创业板
    - 流通市值 30亿~300亿
    - PE 15~60
    - 主力净流入 > 5000万
    """
    print("\n" + "=" * 60)
    print("  第三层：个股精选（龙头共振策略）")
    print("=" * 60)

    # 收集热点板块内的所有股票
    candidate_codes = set()
    for sector in hot_sectors:
        cons = sector["stocks"]
        codes = cons["代码"].tolist()
        candidate_codes.update(codes)

    print(f"  热点板块内股票池: {len(candidate_codes)} 只")

    # 从行情数据筛选基础条件
    if spot_df is None:
        spot_df = ak.stock_zh_a_spot_em()

    df = spot_df[spot_df["代码"].isin(candidate_codes)].copy()

    # 排除科创板/创业板/ST
    df = df[~df["代码"].str.startswith("688")]
    df = df[~df["代码"].str.startswith("30")]
    df = df[~df["名称"].str.contains("ST", case=False, na=False)]
    print(f"  排除科创/创业/ST后: {len(df)}")

    # 涨幅 3%~8%
    df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
    df = df[(df["涨跌幅"] >= 3) & (df["涨跌幅"] <= 8)]
    print(f"  涨幅 3%~8%: {len(df)}")

    # 流通市值 30亿~300亿
    df["流通市值"] = pd.to_numeric(df["流通市值"], errors="coerce")
    df = df[(df["流通市值"] >= 3e9) & (df["流通市值"] <= 3e10)]
    print(f"  流通市值 30亿~300亿: {len(df)}")

    # PE 15~60
    df["市盈率-动态"] = pd.to_numeric(df["市盈率-动态"], errors="coerce")
    df = df[(df["市盈率-动态"] >= 15) & (df["市盈率-动态"] <= 60)]
    print(f"  PE 15~60: {len(df)}")

    if df.empty:
        print("  ❌ 基础条件筛选后无结果")
        return pd.DataFrame()

    # 获取资金流向，筛选主力净流入 > 5000万
    print("  获取资金流向数据...")
    try:
        fund_df = ak.stock_individual_fund_flow_rank(indicator="今日")
        fund_df["今日主力净流入-净额"] = pd.to_numeric(fund_df["今日主力净流入-净额"], errors="coerce")
        fund_positive = set(fund_df[fund_df["今日主力净流入-净额"] > 5e7]["代码"].tolist())
        df = df[df["代码"].isin(fund_positive)]
        print(f"  主力净流入 > 5000万: {len(df)}")
    except Exception as e:
        print(f"  ⚠️ 资金数据获取失败({e}), 跳过")

    if df.empty:
        print("  ❌ 资金条件筛选后无结果")
        return pd.DataFrame()

    # 技术面验证：量能+均线+位置
    print(f"  技术面验证 ({len(df)} 只)...")
    passed = []
    for _, row in df.iterrows():
        code = row["代码"]
        name = row["名称"]
        try:
            hist = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            if hist is None or len(hist) < 60:
                continue

            close = hist["收盘"].astype(float).values
            volume = hist["成交量"].astype(float).values

            # 成交量 > 20日均量 × 1.5
            vol_today = volume[-1]
            vol_ma20 = volume[-20:].mean()
            if vol_today < vol_ma20 * 1.5:
                continue

            # 均线多头: MA5 > MA10 > MA20 > MA60
            ma5 = close[-5:].mean()
            ma10 = close[-10:].mean()
            ma20 = close[-20:].mean()
            ma60 = close[-60:].mean()
            if not (ma5 > ma10 > ma20 > ma60):
                continue

            # 近60日高点附近 (距高点 < 5%)
            high60 = hist["最高"].astype(float).tail(60).max()
            if close[-1] < high60 * 0.95:
                continue

            pct = float(row["涨跌幅"])
            pe = float(row["市盈率-动态"])
            mcap = float(row["流通市值"]) / 1e8

            passed.append({
                "code": code,
                "name": name,
                "pct_change": pct,
                "pe": pe,
                "market_cap_yi": mcap,
                "vol_ratio": vol_today / vol_ma20,
                "dist_high": close[-1] / high60 * 100,
            })
            print(f"  ✓ {code} {name} 涨{pct:.1f}% PE:{pe:.0f} 量比:{vol_today/vol_ma20:.1f}x")
            time.sleep(0.3)
        except Exception:
            continue
        time.sleep(0.2)

    if passed:
        return pd.DataFrame(passed)
    return pd.DataFrame()


# ========== 策略二：启动信号 ==========

def strategy2_breakout(spot_df):
    """
    启动信号策略:
    - 横盘整理 >= 15个交易日
    - 今日放量突破箱体上沿 (量 > 前15日均量 × 2, 收盘 > 近15日最高)
    - 当日涨幅 3%~7%
    - MACD金叉 or DIF从负转正
    - 主力净流入 > 5000万
    """
    print("\n" + "=" * 60)
    print("  策略二：启动信号（横盘突破）")
    print("=" * 60)

    if spot_df is None:
        spot_df = ak.stock_zh_a_spot_em()

    df = spot_df.copy()
    # 基础过滤
    df = df[~df["代码"].str.startswith("688")]
    df = df[~df["代码"].str.startswith("30")]
    df = df[~df["名称"].str.contains("ST", case=False, na=False)]
    df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
    df["流通市值"] = pd.to_numeric(df["流通市值"], errors="coerce")
    df["市盈率-动态"] = pd.to_numeric(df["市盈率-动态"], errors="coerce")

    # 涨幅 3%~7%
    df = df[(df["涨跌幅"] >= 3) & (df["涨跌幅"] <= 7)]
    print(f"  涨幅 3%~7%: {len(df)}")

    # 流通市值/PE过滤 (放宽一些)
    df = df[(df["流通市值"] >= 2e9) & (df["流通市值"] <= 5e10)]
    df = df[df["市盈率-动态"] > 0]
    print(f"  市值+PE过滤后: {len(df)}")

    # 资金验证
    try:
        fund_df = ak.stock_individual_fund_flow_rank(indicator="今日")
        fund_df["今日主力净流入-净额"] = pd.to_numeric(fund_df["今日主力净流入-净额"], errors="coerce")
        fund_positive = set(fund_df[fund_df["今日主力净流入-净额"] > 5e7]["代码"].tolist())
        df = df[df["代码"].isin(fund_positive)]
        print(f"  主力净流入 > 5000万: {len(df)}")
    except Exception as e:
        print(f"  ⚠️ 资金数据获取失败({e})")

    if df.empty:
        print("  ❌ 基础筛选无结果")
        return pd.DataFrame()

    # 限制检查数量
    df = df.head(80)
    print(f"  逐只检查突破信号 ({len(df)} 只)...")

    passed = []
    for _, row in df.iterrows():
        code = row["代码"]
        name = row["名称"]
        try:
            hist = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            if hist is None or len(hist) < 30:
                continue

            close = hist["收盘"].astype(float).values
            volume = hist["成交量"].astype(float).values
            high = hist["最高"].astype(float).values

            # 前15日是否横盘 (振幅 < 15%)
            prev_close = close[-16:-1]
            prev_high = high[-16:-1]
            box_high = prev_high.max()
            box_low = prev_close.min()
            box_range = (box_high - box_low) / box_low * 100
            if box_range > 15:
                continue

            # 今日收盘突破箱体上沿
            if close[-1] <= box_high:
                continue

            # 放量: 今日量 > 前15日均量 × 2
            vol_today = volume[-1]
            vol_avg15 = volume[-16:-1].mean()
            if vol_today < vol_avg15 * 2:
                continue

            # MACD: DIF从负转正 or 金叉
            ema12 = pd.Series(close).ewm(span=12).mean()
            ema26 = pd.Series(close).ewm(span=26).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9).mean()

            dif_today = dif.iloc[-1]
            dif_yesterday = dif.iloc[-2]
            dea_today = dea.iloc[-1]
            dea_yesterday = dea.iloc[-2]

            macd_signal = False
            # DIF从负转正
            if dif_yesterday < 0 and dif_today > 0:
                macd_signal = True
            # 金叉 (DIF上穿DEA)
            if dif_yesterday <= dea_yesterday and dif_today > dea_today:
                macd_signal = True
            # DIF>0且向上也算
            if dif_today > 0 and dif_today > dif_yesterday:
                macd_signal = True

            if not macd_signal:
                continue

            pct = float(row["涨跌幅"])
            pe = float(row["市盈率-动态"])
            mcap = float(row["流通市值"]) / 1e8

            passed.append({
                "code": code,
                "name": name,
                "pct_change": pct,
                "pe": pe,
                "market_cap_yi": mcap,
                "vol_ratio": vol_today / vol_avg15,
                "box_range": box_range,
                "breakout_pct": (close[-1] - box_high) / box_high * 100,
            })
            print(f"  ✓ {code} {name} 突破! 涨{pct:.1f}% 量比:{vol_today/vol_avg15:.1f}x 箱体振幅:{box_range:.1f}%")
            time.sleep(0.3)
        except Exception:
            continue
        time.sleep(0.2)

    if passed:
        return pd.DataFrame(passed)
    return pd.DataFrame()


# ========== 综合评分 & 输出 ==========

def score_strategy1(df):
    """龙头共振评分"""
    if df.empty:
        return df
    df = df.copy()
    df["score"] = 0.0
    # 涨幅适中 (4-6%最佳)
    df["score"] += df["pct_change"].apply(lambda x: 25 if 4 <= x <= 6 else 15)
    # 量比越大越好
    df["score"] += df["vol_ratio"].apply(lambda x: min(x / 3 * 25, 25))
    # 距高点越近越好
    df["score"] += df["dist_high"].apply(lambda x: 25 if x >= 98 else (20 if x >= 95 else 15))
    # PE合理性
    df["score"] += df["pe"].apply(lambda x: 25 if 20 <= x <= 40 else 15)
    return df.sort_values("score", ascending=False)


def score_strategy2(df):
    """启动信号评分"""
    if df.empty:
        return df
    df = df.copy()
    df["score"] = 0.0
    # 量比越大越好 (突破确认)
    df["score"] += df["vol_ratio"].apply(lambda x: min(x / 4 * 30, 30))
    # 箱体越窄越好 (能量积蓄越久)
    df["score"] += df["box_range"].apply(lambda x: 30 if x <= 8 else (20 if x <= 12 else 10))
    # 涨幅适中
    df["score"] += df["pct_change"].apply(lambda x: 20 if 4 <= x <= 6 else 15)
    # 突破幅度
    df["score"] += df["breakout_pct"].apply(lambda x: 20 if x >= 2 else 10)
    return df.sort_values("score", ascending=False)


def print_results(title, df, strategy_type):
    """输出结果"""
    print("\n" + "=" * 60)
    print(f"  🎯 {title}")
    print("=" * 60)

    if df.empty:
        print("  本策略今日无符合条件标的")
        return

    df = df.head(10)
    for i, (_, row) in enumerate(df.iterrows()):
        tag = "⭐" if row["score"] >= 80 else ("✅" if row["score"] >= 60 else "👀")
        print(f"  {tag} {i+1}. {row['code']} {row['name']}")
        if strategy_type == 1:
            print(f"     涨幅:{row['pct_change']:.1f}%  PE:{row['pe']:.0f}  量比:{row['vol_ratio']:.1f}x  距高点:{row['dist_high']:.1f}%  评分:{row['score']:.0f}")
        else:
            print(f"     涨幅:{row['pct_change']:.1f}%  PE:{row['pe']:.0f}  量比:{row['vol_ratio']:.1f}x  箱体:{row['box_range']:.1f}%  评分:{row['score']:.0f}")


def parse_args():
    parser = argparse.ArgumentParser(description="高胜率选股系统")
    parser.add_argument(
        "--min-sh-pct", type=float, default=0.0,
        help="沪指最低涨幅要求 (%%), 默认 0.0 (即要求 > 0%%)"
    )
    parser.add_argument(
        "--no-up-majority", action="store_true",
        help="跳过「上涨家数>下跌家数」检查"
    )
    parser.add_argument(
        "--min-north", type=float, default=0.0,
        help="北向资金最低净流入 (亿元), 默认 0"
    )
    parser.add_argument(
        "--skip-market-check", action="store_true",
        help="跳过第一层市场环境检查, 强制执行选股"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(f"  高胜率选股系统 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  策略一: 龙头共振 | 策略二: 启动信号")
    print("=" * 60)

    # 第一层: 市场环境
    if args.skip_market_check:
        print("\n  ⚠️ 已跳过市场环境检查 (--skip-market-check)")
        spot_df = ak.stock_zh_a_spot_em()
        spot_df["涨跌幅"] = pd.to_numeric(spot_df["涨跌幅"], errors="coerce")
        market_ok = True
        ctx = {"spot_df": spot_df}
    else:
        market_ok, ctx = check_market_environment(
            min_sh_pct=args.min_sh_pct,
            require_up_majority=not args.no_up_majority,
            min_north_net=args.min_north * 1e8,
        )

    if not market_ok:
        print("\n" + "=" * 60)
        print("  📊 结论: 市场环境不满足, 建议空仓观望")
        print("  💡 可用 --skip-market-check 强制跳过, 或调整阈值:")
        print(f"     --min-sh-pct -0.5    (允许沪指微跌)")
        print(f"     --no-up-majority     (不要求涨多跌少)")
        print(f"     --min-north -10      (允许北向流出10亿)")
        print("=" * 60)
        return

    spot_df = ctx.get("spot_df")

    # ========== 策略一: 龙头共振 ==========
    # 第二层: 板块共振
    hot_sectors = find_hot_sectors()

    s1_result = pd.DataFrame()
    if hot_sectors:
        # 第三层+第四层: 个股精选+资金验证
        s1_result = strategy1_stock_filter(hot_sectors, spot_df)
        if not s1_result.empty:
            s1_result = score_strategy1(s1_result)

    # ========== 策略二: 启动信号 ==========
    s2_result = strategy2_breakout(spot_df)
    if not s2_result.empty:
        s2_result = score_strategy2(s2_result)

    # ========== 输出结果 ==========
    print_results("策略一：龙头共振 推荐", s1_result, 1)
    print_results("策略二：启动信号 推荐", s2_result, 2)

    # 综合推荐
    print("\n" + "=" * 60)
    print("  📋 买入执行规则提醒")
    print("=" * 60)
    print("  • 买入时机: 14:00~14:45 确认尾盘不跌再买")
    print("  • 单只仓位: ≤ 总仓位 20%")
    print("  • 分批操作: 先买 50%, 次日确认再加仓")
    print()
    print("  📋 止损规则")
    print("  • 跌破当日最低价 → 立即止损")
    print("  • 次日低开 > 3% → 止损")
    print("  • 板块当日跌 > 1% → 止损")
    print("  • 持有 3 日未涨 → 止损或减仓")
    print()
    print("  📋 止盈规则")
    print("  • 涨幅 > 10% → 卖出 50%")
    print("  • 涨停次日高开低走 → 卖出")
    print("  • 缩量上涨 → 减仓")
    print("  • 大盘转弱 → 无条件减仓")
    print()
    print("  ⚠️  免责声明: 本策略仅供学习参考, 不构成投资建议。")
    print("      股市有风险, 投资需谨慎。")


if __name__ == "__main__":
    main()
