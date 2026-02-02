import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime

st.set_page_config(
    page_title="Kakao 2025 주가 분석 대시보드",
    layout="wide"
)

# -------------------------------
# 0. 데이터 로딩 함수
# -------------------------------
@st.cache_data
def load_price_data(ticker: str, start: str, end: str):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    return df

def compute_technical_indicators(df: pd.DataFrame):
    df = df.copy()
    # 이동평균
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA60"] = df["Close"].rolling(window=60).mean()
    df["MA120"] = df["Close"].rolling(window=120).mean()
    # 볼린저 밴드 (20일, 2표준편차)
    rolling_std = df["Close"].rolling(window=20).std()
    df["BB_MID"] = df["MA20"]
    df["BB_UPPER"] = df["BB_MID"] + 2 * rolling_std
    df["BB_LOWER"] = df["BB_MID"] - 2 * rolling_std
    # 일일 수익률
    df["Return"] = df["Close"].pct_change()
    return df

def compute_risk_metrics(df: pd.DataFrame):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    # 변동성(연율화 X, 일간)
    daily_vol = df["Return"].std()
    # 최대 낙폭(MDD)
    cum_max = df["Close"].cummax()
    drawdown = df["Close"] / cum_max - 1.0
    mdd = drawdown.min()
    # VaR (95%, 99%)
    var_95 = np.percentile(df["Return"].dropna(), 5)
    var_99 = np.percentile(df["Return"].dropna(), 1)
    return {
        "daily_vol": daily_vol,
        "mdd": mdd,
        "var_95": var_95,
        "var_99": var_99,
        "returns_df": df
    }

def detect_large_volume_moves(df: pd.DataFrame, ret_q=90, vol_q=90):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)

    ret_threshold_up = np.percentile(df["Return"], ret_q)
    ret_threshold_down = np.percentile(df["Return"], 100 - ret_q)
    vol_threshold = np.percentile(df["Volume"], vol_q)

    cond_up = (df["Return"] >= ret_threshold_up) & (df["Volume"] >= vol_threshold)
    cond_down = (df["Return"] <= ret_threshold_down) & (df["Volume"] >= vol_threshold)

    large_up = df[cond_up]
    large_down = df[cond_down]

    return large_up, large_down, {
        "ret_q": ret_q,
        "vol_q": vol_q,
        "ret_threshold_up": ret_threshold_up,
        "ret_threshold_down": ret_threshold_down,
        "vol_threshold": vol_threshold
    }

def detect_support_resistance(df: pd.DataFrame, bins=20, min_touches=3):
    """
    매우 단순한 방식:
    - 종가를 일정 구간으로 binning
    - 많이 등장한 가격대(횟수 >= min_touches)를 지지/저항 후보로 사용
    """
    prices = df["Close"].dropna()
    counts, bin_edges = np.histogram(prices, bins=bins)
    levels = []

    for i, c in enumerate(counts):
        if c >= min_touches:
            level = (bin_edges[i] + bin_edges[i+1]) / 2
            levels.append(level)

    # 중복/인접 레벨 간단히 정리
    levels = sorted(levels)
    merged_levels = []
    if levels:
        current = levels[0]
        for lvl in levels[1:]:
            if abs(lvl - current) / current < 0.02:  # 2% 이내면 합침
                current = (current + lvl) / 2
            else:
                merged_levels.append(current)
                current = lvl
        merged_levels.append(current)

    return merged_levels

# -------------------------------
# 1. 사이드바 설정
# -------------------------------
st.sidebar.title("설정")

st.sidebar.markdown("**티커(Ticker)**")
ticker = st.sidebar.text_input("카카오 KOSPI 티커 (기본: 035720.KS)", value="035720.KS")

st.sidebar.markdown("**분석 기간 (2025년 중심)**")
start_date = st.sidebar.date_input("시작일", value=datetime(2025, 1, 1))
end_date = st.sidebar.date_input("종료일", value=datetime(2025, 12, 31))

st.sidebar.markdown("**거래량-가격 필터 기준**")
ret_q = st.sidebar.slider("수익률 분위수 (상/하위)", min_value=70, max_value=99, value=90, step=1)
vol_q = st.sidebar.slider("거래량 분위수", min_value=70, max_value=99, value=90, step=1)

st.sidebar.markdown("**지지/저항선 탐지 설정**")
bins = st.sidebar.slider("가격 구간(bin) 개수", min_value=10, max_value=100, value=30, step=5)
min_touches = st.sidebar.slider("최소 터치 횟수", min_value=2, max_value=10, value=3, step=1)

st.sidebar.markdown("---")
st.sidebar.info("※ 실제 투자 결정 전 반드시 본인 판단과 추가 검증이 필요합니다.")

# -------------------------------
# 2. 데이터 로딩
# -------------------------------
st.title("📊 카카오(Kakao) 2025 주가 분석 대시보드")

st.write(f"분석 티커: **{ticker}**, 기간: **{start_date} ~ {end_date}**")

if start_date >= end_date:
    st.error("시작일이 종료일보다 같거나 이후입니다. 기간을 다시 설정해주세요.")
    st.stop()

df = load_price_data(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

if df.empty:
    st.error("해당 기간에 대한 데이터가 없습니다. 티커 또는 기간을 확인해주세요.")
    st.stop()

df = compute_technical_indicators(df)

# -------------------------------
# 3. 탭 구성
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① 추세 & 변동성 (MA & Bollinger)",
    "② 거래량-가격 상관관계",
    "③ 수익률 분포 & 리스크 (VaR/MDD)",
    "④ 캔들 & 지지/저항선",
    "⑤ 시계열 분해 (Trend/Seasonality)"
])

# -------------------------------
# 3-1. 추세 & 변동성 (MA & Bollinger)
# -------------------------------
with tab1:
    st.subheader("① 이동평균선 & 볼린저 밴드를 통한 추세·변동성 분석")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", name="종가",
        line=dict(color="black", width=1)
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA20"],
        mode="lines", name="MA 20",
        line=dict(color="blue", width=1)
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA60"],
        mode="lines", name="MA 60",
        line=dict(color="orange", width=1)
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA120"],
        mode="lines", name="MA 120",
        line=dict(color="green", width=1)
    ))

    # 볼린저 밴드
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_UPPER"],
        line=dict(color="rgba(173,216,230,0.8)", width=1),
        name="Bollinger Upper",
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_LOWER"],
        line=dict(color="rgba(173,216,230,0.8)", width=1),
        fill='tonexty',
        fillcolor='rgba(173,216,230,0.2)',
        name="Bollinger Lower",
        showlegend=True
    ))

    fig.update_layout(
        height=600,
        xaxis_title="날짜",
        yaxis_title="가격 (KRW)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **해석 가이드**
    - **MA 골든/데드 크로스**: 단기(MA20)가 중기·장기(MA60, MA120)를 상향 돌파하면 매수 우위 신호, 반대는 매도 우위 신호로 볼 수 있습니다.
    - **볼린저 밴드 상단/하단 접촉**: 상단 밴드 근처는 과열 가능성, 하단 밴드는 과매도 구간일 수 있습니다.
    - 밴드 폭이 넓어지는 시점은 **변동성 확산 시기**로, 리스크 관리가 특히 중요합니다.
    """)

# -------------------------------
# 3-2. 거래량-가격 상관관계
# -------------------------------
with tab2:
    st.subheader("② 거래량(Volume)과 가격의 상관관계 분석")

    df_vol = df.copy()
    df_vol["Return"] = df_vol["Close"].pct_change()
    df_vol.dropna(inplace=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        # 가격 & 거래량 (이중축)
        fig_vol = make_fig = go.Figure()

        fig_vol.add_trace(go.Scatter(
            x=df_vol.index, y=df_vol["Close"],
            mode="lines", name="종가",
            line=dict(color="black")
        ))

        fig_vol.add_trace(go.Bar(
            x=df_vol.index, y=df_vol["Volume"],
            name="거래량",
            marker_color="rgba(0, 0, 255, 0.4)",
            yaxis="y2"
        ))

        fig_vol.update_layout(
            height=600,
            xaxis=dict(domain=[0.0, 1.0]),
            yaxis=dict(title="가격 (KRW)"),
            yaxis2=dict(
                title="거래량",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            template="plotly_white",
            barmode="overlay"
        )

        st.plotly_chart(fig_vol, use_container_width=True)

    with col2:
        corr = df_vol[["Return", "Volume"]].corr().iloc[0, 1]
        st.metric("수익률-거래량 상관계수", f"{corr:.3f}")
        st.caption("※ 단순 상관계수로, 특정 이벤트일 분석 시 더 정교한 접근이 필요합니다.")

    # 대량 거래 동반 상승/하락 일자 추출
    large_up, large_down, info = detect_large_volume_moves(df.copy(), ret_q=ret_q, vol_q=vol_q)

    st.markdown(f"""
    **대량 거래 기준**
    - 수익률 상/하위 분위수: **{info['ret_q']}%**
    - 거래량 상위 분위수: **{info['vol_q']}%**
    """)

    col_up, col_down = st.columns(2)

    with col_up:
        st.markdown("**📈 대량 거래 동반 상승 일자**")
        if large_up.empty:
            st.write("조건에 해당하는 상승 구간이 없습니다.")
        else:
            st.dataframe(
                large_up[["Close", "Volume", "Return"]]
                .assign(Return=lambda x: x["Return"] * 100)
                .rename(columns={"Close": "종가", "Volume": "거래량", "Return": "수익률(%)"})
            )

    with col_down:
        st.markdown("**📉 대량 거래 동반 하락 일자**")
        if large_down.empty:
            st.write("조건에 해당하는 하락 구간이 없습니다.")
        else:
            st.dataframe(
                large_down[["Close", "Volume", "Return"]]
                .assign(Return=lambda x: x["Return"] * 100)
                .rename(columns={"Close": "종가", "Volume": "거래량", "Return": "수익률(%)"})
            )

    st.markdown("""
    **해석 가이드**
    - **가격 상승 + 대량 거래**: 새로운 추세 시작/강화 가능성을 시사할 수 있습니다.
    - **가격 하락 + 대량 거래**: 투매·공포가 집중된 구간일 수 있으며, 이후 반등 포인트가 되기도 합니다.
    - 이벤트(실적 발표, 규제 뉴스 등) 시점을 함께 마킹하면 원인-결과 관계를 더 명확히 볼 수 있습니다.
    """)

# -------------------------------
# 3-3. 수익률 분포 & 리스크 (VaR/MDD)
# -------------------------------
with tab3:
    st.subheader("③ 일일 수익률 분포 및 리스크 측정 (VaR, 변동성, MDD)")

    risk = compute_risk_metrics(df.copy())
    returns_df = risk["returns_df"]

    col_l, col_r = st.columns(2)

    with col_l:
        st.metric("일간 변동성 (표준편차)", f"{risk['daily_vol'] * 100:.2f}%")
        st.metric("최대 낙폭 (MDD)", f"{risk['mdd'] * 100:.2f}%")

    with col_r:
        st.metric("VaR 95% (일간)", f"{risk['var_95'] * 100:.2f}%")
        st.metric("VaR 99% (일간)", f"{risk['var_99'] * 100:.2f}%")

    # 히스토그램 (matplotlib 사용)
    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.hist(returns_df["Return"], bins=40, color="skyblue", edgecolor="black", alpha=0.7)
    ax.axvline(risk["var_95"], color="red", linestyle="--", label=f"VaR 95% ({risk['var_95']*100:.2f}%)")
    ax.axvline(risk["var_99"], color="purple", linestyle="--", label=f"VaR 99% ({risk['var_99']*100:.2f}%)")
    ax.set_title("일일 수익률 히스토그램")
    ax.set_xlabel("일일 수익률")
    ax.set_ylabel("빈도")
    ax.legend()
    st.pyplot(fig2)

    # 최대 낙폭 시각화
    st.markdown("**최대 낙폭(MDD) 구간 시각화**")
    cum_max = returns_df["Close"].cummax()
    drawdown = returns_df["Close"] / cum_max - 1.0
    mdd_idx = drawdown.idxmin()
    mdd_end_price = returns_df.loc[mdd_idx, "Close"]
    mdd_start_idx = (returns_df["Close"][:mdd_idx]).idxmax()
    mdd_start_price = returns_df.loc[mdd_start_idx, "Close"]

    fig_mdd = go.Figure()
    fig_mdd.add_trace(go.Scatter(
        x=returns_df.index, y=returns_df["Close"],
        mode="lines", name="종가"
    ))
    fig_mdd.add_trace(go.Scatter(
        x=[mdd_start_idx, mdd_end_price and mdd_idx],
        y=[mdd_start_price, mdd_end_price],
        mode="lines+markers",
        line=dict(color="red", width=2, dash="dot"),
        marker=dict(size=8),
        name="MDD 구간"
    ))
    fig_mdd.update_layout(
        height=400,
        xaxis_title="날짜",
        yaxis_title="가격 (KRW)",
        template="plotly_white"
    )
    st.plotly_chart(fig_mdd, use_container_width=True)

    st.markdown("""
    **해석 가이드**
    - 히스토그램이 좌측 꼬리가 두꺼우면 **하방 리스크가 집중**되었음을 의미합니다.
    - VaR 95%: 통계적으로 하루에 이 값보다 더 크게 손실 볼 확률이 약 5%라는 의미입니다.
    - MDD 구간은 해당 연도 중 가장 뼈아팠던 **'검은 날'**들을 시각적으로 보여줍니다.
    """)

# -------------------------------
# 3-4. 캔들차트 & 지지/저항선
# -------------------------------
with tab4:
    st.subheader("④ 캔들차트 기반 주요 지지/저항선 자동 탐지")

    levels = detect_support_resistance(df.copy(), bins=bins, min_touches=min_touches)

    fig_candle = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Kakao"
            )
        ]
    )

    # 지지/저항선 수평선 추가
    for lvl in levels:
        fig_candle.add_hline(
            y=lvl,
            line_dash="dot",
            line_color="blue",
            opacity=0.4
        )

    fig_candle.update_layout(
        height=600,
        xaxis_title="날짜",
        yaxis_title="가격 (KRW)",
        template="plotly_white"
    )

    st.plotly_chart(fig_candle, use_container_width=True)

    if levels:
        st.markdown("**탐지된 지지/저항 가격대 (단순 빈도 기반, KRW)**")
        st.write([round(l, -2) for l in levels])  # 100원 단위로 반올림
    else:
        st.write("설정 조건에 해당하는 지지/저항 후보 레벨이 없습니다. bins/최소 터치 횟수를 조정해보세요.")

    st.markdown("""
    **해석 가이드**
    - 위의 수평선은 **가격이 여러 번 머물렀던 구간**을 단순히 빈도 기반으로 추출한 것입니다.
    - 반복적으로 상단을 막았던 가격대는 **심리적 저항선**, 하단을 받쳐준 가격대는 **심리적 지지선**일 수 있습니다.
    - 2025년의 이런 레벨들은 2026년 매매 전략에서 손절/익절 레벨 가이드로 활용할 수 있습니다.
    """)

# -------------------------------
# 3-5. 시계열 분해 (Trend, Seasonality, Residual)
# -------------------------------
with tab5:
    st.subheader("⑤ 시계열 분해 분석 (추세, 계절성, 잔차)")

    # 시계열 분해: 1년 데이터라면 월간 패턴을 보기 위해 period=21(거래일 기준 약 한 달) 정도 사용
    ts = df["Close"].asfreq("B")  # Business day frequency
    ts_interpolated = ts.interpolate()  # 결측 보정

    try:
        decomposition = seasonal_decompose(ts_interpolated, model="additive", period=21)
        comp_df = pd.DataFrame({
            "Observed": decomposition.observed,
            "Trend": decomposition.trend,
            "Seasonal": decomposition.seasonal,
            "Residual": decomposition.resid
        })

        fig_dec = go.Figure()

        for i, col in enumerate(["Observed", "Trend", "Seasonal", "Residual"]):
            fig_dec.add_trace(go.Scatter(
                x=comp_df.index,
                y=comp_df[col],
                name=col,
                visible=True if col == "Observed" else "legendonly"
            ))

        fig_dec.update_layout(
            height=600,
            xaxis_title="날짜",
            template="plotly_white"
        )

        st.plotly_chart(fig_dec, use_container_width=True)

        st.markdown("""
        **해석 가이드**
        - **Trend(추세)**: 노이즈를 제거한 카카오 주가의 본질적 방향성을 보여줍니다.
        - **Seasonal(계절성)**: 특정 월·분기·이벤트(실적 발표, 연말 등) 주변에서 반복되는 패턴을 포착합니다.
        - **Residual(잔차)**: 추세·계절성으로 설명되지 않는 우발적 요인(뉴스, 공매도 급증 등)에 해당합니다.
        """)
    except Exception as e:
        st.error(f"시계열 분해 과정에서 오류가 발생했습니다: {e}")
        st.caption("데이터 길이가 너무 짧거나 결측치가 많은 경우 발생할 수 있습니다.")

# -------------------------------
# 페이지 하단 안내
# -------------------------------
st.markdown("---")
st.markdown("""
**면책 조항 (Disclaimer)**  
본 대시보드는 과거 데이터에 기반한 정량적 분석 도구이며,  
**어떠한 투자 수익도 보장하지 않으며 투자 손실에 대한 책임은 전적으로 투자자 본인에게 있습니다.**
""")