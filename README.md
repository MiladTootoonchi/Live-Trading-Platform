<h1 align = "center"> Live-Trading-Platform </h1>

***

**Year:** 2025

**Team members:** \
Milad Tootoonchi \
Makka Dulgaeva

***

<h2 align = "center"> Introduction </h2>

The financial markets operate as a vast and intricate system, where millions of traders make decisions in real time, striving to outperform one another. Historically, human intuition and experience dominated this space. Stock exchanges were filled with traders engaging in open outcry, making rapid decisions based on their expertise and instincts. 

However, the landscape has undergone a significant transformation. Human traders are no longer the primary force driving the markets. Instead, algorithmic trading systems have taken precedence operating silently, with extraordinary speed and efficiency. These systems analyze vast quantities of data within milliseconds, executing trades far beyond human capability. 

Despite their perceived complexity, trading algorithms are not a product of chance. They are meticulously designed by financial professionals and engineers, programmed to follow specific strategies and navigate market fluctuations with precision. This project serves as an opportunity to understand the mechanisms behind algorithmic trading, how these systems function, how they execute decisions, and how one can develop a fundamental algorithmic trading model. 

Since the inception of commerce, societies have engaged in trade, progressing from ancient marketplaces to highly sophisticated stock exchanges. Today, finance is experiencing a paradigm shift, largely driven by advancements in technology, changes that occur beyond the immediate perception of most individuals. 

Financial Technology (FinTech) has revolutionized the management of financial transactions, encompassing innovations such as online banking, digital payments, and cryptocurrencies. At the forefront of this revolution is algorithmic trading, wherein computational models, rather than human judgment, dictate trading decisions. 

Modern trading success is no longer solely reliant on human intuition; rather, it is increasingly determined by data analysis, strategic modeling, and automation. This project presents an opportunity to engage with algorithmic trading by designing and testing a Python-based trading model, one that makes objective, data-driven decisions, free from emotional bias. 

***

<h2 align = "center"> Background </h2>

A useful analogy for algorithmic trading is that of a predator in its natural habitat, constantly surveying its surroundings, detecting patterns, and responding with precision. Similarly, an algorithmic trading system continuously monitors market trends, identifies potential opportunities, and executes transactions at high velocity. 

These systems operate around the clock, analyzing stocks, cryptocurrencies, and other financial instruments, identifying price discrepancies that may yield profit. However, despite their efficiency, success in algorithmic trading is not guaranteed. Financial markets are inherently unpredictable, requiring continuous adaptation. A strategy that proves effective today may become obsolete in the future. 

Thus, algorithmic trading systems must undergo regular refinement and optimization, evolving in response to shifting market conditions, much like any adaptive system seeking longterm viability in a competitive environment. This project will allow us to step into this world, to create a trading algorithm and test its survival in a simulated market, where only the most effective strategies endure. 

<h3 align = "center"> Problem Statement </h3>

Traditional trading methods often depend on manual decision-making processes, which are both time-intensive and prone to human error. In contrast, algorithmic trading utilizes data-driven strategies to improve execution speed, efficiency, and consistency. However, the development of a robust trading algorithm necessitates a comprehensive understanding of several key areas: 

- Financial markets and trading strategies 

- Market data processing and analysis 

- Automated trade execution using APIs 

- Risk management and performance evaluation 

To bridge the gap between theoretical knowledge and practical application, this project aims to develop a Python-based trading bot capable of analysing market trends, executing trades, and assessing strategy performance within a simulated paper trading environment.

The overall goal is to develop an automated trading system.  
In order to achieve this the following objectives and activities has been set: \

1. Research 

    - Conduct a theoretical review of key concepts, including: 
        - Introduction to FinTech 
        - Algorithmic Trading and Market Access 
        - Trading Strategies and Back testing 
        - Introduction to Python for Trading 

    - Identify state-of-the-art open-source solutions through a technical review. 
    - Summarize findings and propose a project specification. 

2. Development/Method 

    - Develop a system for deploying a trading strategy based on selected indicators and predefined rules. 
        - Propose system architecture  
        - Data pipeline- Retrieve, clean, and structure real-time and historical market data using Alpaca’s API. 
        - Development of algorithm  
        - Implement a back testing framework to evaluate the strategy’s effectiveness using historical data. 

    - Deploy the strategy in a simulated trading environment using Alpaca’s paper-trading API. 
    - Summarize findings and insights from the development process. 

3. Evaluation 

    - Suggest a test plan which measure return, risk, and stability based on back testing results and compare with Alpaca’s paper-trading performance. 
        - Analyse the strategy’s robustness by testing it under different market conditions and evaluating its sensitivity to parameter changes. 
        - Assess execution speed and identify potential technical bottlenecks. 
        - Assess key performance metrics, including profit/loss, drawdown, and risk management. 
    - Summarize evaluation results and insights. 

4. Recommendations and Further Work 

    - Provide suggestions for improving the trading strategy based on evaluation results. 
    - Identify areas for future research and potential enhancements, such as alternative indicators, improved risk management, or AI-based optimizations. 

5. Limitations   

    - The project relies on Alpaca’s API for paper trading, which doesn’t account for real-world factors like slippage or liquidity issues. 
    - The use of Python limits performance due to its relative speed compared to other programming languages. 
    - The project focuses mainly on technical indicators, which may not adapt well to changing market conditions. 
    - The back testing framework is based on historical data, which may not predict future market behaviour accurately. 
    - Risk management is limited by predefined parameters and lacks exposure to real capital. 
    - The project refrain from utilizing the shorter strategy in our approach. 

***

<h2 align = "center"> Theory & Key Concepts </h2>

This chapter provides an overview of the key concepts essential for developing an automated trading system. It covers topics such as FinTech, algorithmic trading, trading strategies, and back testing. Additionally, it introduces Python’s role in implementing trading strategies and interacting with platforms like Alpaca. Understanding these concepts will lay the foundation for creating and evaluating a successful trading strategy. 

<h2 align = "center"> What is FinTech? </h2>

Financial technology, or fintech, refers to the use of innovative technologies to deliver and improve financial services. From mobile banking and digital wallets to blockchain and algorithmic trading, fintech is transforming how individuals, businesses, and institutions interact with money. 

Driven by advancements in software, data analytics, and connectivity, fintech has disrupted traditional banking models by offering faster, cheaper, and more accessible solutions. It enables everything from peer-to-peer payments and robo-advisors to crowdfunding platforms and decentralized finance (DeFi).

![Fintech (financial technology) and the European Union: State of play and  outlook | Epthinktank | European Parliament](https://i0.wp.com/epthinktank.eu/wp-content/uploads/2019/02/eprs-briefing-635513-fintech-and-eu-final.jpg?fit=1000%2C306&ssl=1 "FinTech")

At its core, fintech blends finance and technology to increase efficiency, enhance customer experiences, and open up new opportunities for financial inclusion across the globe. As digital adoption continues to rise, fintech is reshaping the future of finance—making it more agile, intelligent, and customer-focused than ever before.

“Algorithmic trading is a process for executing orders utilizing automated and pre-programmed trading instructions to account for variables such as price, timing and volume.

![Algorithmic Trading: Definition, How It Works, Pros & Cons](https://www.investopedia.com/thmb/j0RDfj9IIW_pSffoBUikqTyjs8U=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/dotdash_Final_Algorithmic_Trading_Apr_2020-01-59aa25326afd47edb2e847c0e18f8ce2.jpg)

An algorithm is a set of directions for solving a problem. Computer algorithms send small portions of the full order to the market over time.” - Investopedia. 

<h2 align = "center"> Introduction to Alpaca </h2>
In today’s digital-first economy, efficient and flexible market access is a critical component for individual and institutional investors alike. Market access refers to the ability to interact with and trade within financial markets, including stock exchanges, forex, and derivatives markets. The evolution of financial technology has enabled new forms of market access-through APIs, low-latency trading platforms, and algorithmic interfaces—empowering developers and traders to automate strategies and engage with global markets in real-time. 

One of the prominent platforms facilitating this innovation is Alpaca Markets. Alpaca is a modern commission-free brokerage platform that offers robust APIs for trading U.S. stocks and ETFs. Designed with developers in mind, Alpaca provides real-time market data, paper trading environments, and order execution capabilities through simple REST and WebSocket interfaces. Its emphasis on algorithmic and programmatic trading makes it an attractive solution for fintech startups, quantitative traders, and academic researchers exploring financial automation. 

![Alpaca Launches Next-Gen Order Management System That Makes Order  Processing 100x Faster](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTgxUaS1mTtNlVD2XAtydSKXtWsgAGIhygQ-A&s "Alpaca API")

By lowering the barriers to entry, Alpaca represents a shift toward democratizing financial markets—offering accessible, scalable, and customizable trading infrastructure. This project explores the fundamentals of market access and highlights how Alpaca Markets enables seamless integration of trading algorithms, portfolio management, and data-driven investment strategies. 

<h2 align = "center"> Concepts for Strategies </h2>

The decision to buy or sell financial securities uses a predetermined algorithm, with the intent to make a profit. In this platform, this algorithm is a python script which can take historical data as an input and give a decision to buy / sell / hold as output. 

In trading, there are several key strategies that guide decision-making. These strategies can be automated through algorithmic trading systems, allowing for faster and more efficient execution of trades based on specific signals and market conditions. 

- Momentum: This strategy involves buying stocks that are trending upward or selling those that are trending downward, with the expectation that these trends will continue. Traders rely on the idea that strong trends often persist for a period. 

- Mean Reversion: Traders using this strategy believe that stock prices will eventually return to their historical average. As a result, they buy stocks that are undervalued and sell those that are overpriced, expecting price corrections. 

- Moving Average Strategies: Comparing the stock’s current price with its moving averages (20-day, 50-day, 200-day) to generate buy/sell signals. 

- RSI (Relative Strength Index): Used to determine whether a stock is overbought (RSI > 70) or oversold (RSI < 30), providing potential buy/sell signals.

- MACD (Moving Average Convergence Divergence): A momentum indicator based on the crossover of the MACD line and the signal line, used for buy/sell decisions.

- Bollinger Bands: Utilized to measure stock volatility and potential price movements based on the distance between the upper and lower bands.

- Rule based strategy: A simple strategy where the program sends buy / sell / hold signals based on close-price thesholds. Only position data is needed here.

- Neural nettworks: A neural network is a type of machine learning model inspired by the human brain, consisting of interconnected nodes, or "neurons," arranged in layers. These networks learn from data by processing information and adjusting the strength of connections between neurons to recognize patterns, make predictions, and solve complex problems. 
    - RNN (LSTM): A recurrent neural network (RNN) is a type of neural network with a feedback loop that allows it to process sequential data by using past information to influence current outputs. A Long Short-Term Memory (LSTM) is a specialized type of RNN designed to handle long-term dependencies in data more effectively, using memory cells with gates to control the flow of information over time. 

***
<h2 align = "center"> Program Design </h2>

kart og forklaring over hvordan systemer samkjører

kart og forklaring over hvordan pakkene samkjører

kart og forkalring over maskinlærepakken

<h3 align = "center"> Data and Data Collection </h3>

Description of the dataset(s)

Data sources

Data quality issues

Preprocessing steps (cleaning, transformations, feature engineering)

Challenges and constraints

<h3 align = "center"> Manual (How to use the program) </h3>

Førsteg: hva må man gjøre før man starter den
hvordan starte / bruke den


***
<h2 align = "center"> Results </h2>

...

<h3 align = "center"> Backtesting Results </h3>

strategier

maskinlære

***

<h2 align = "center"> Discussion & Future Work </h2>

diskusjon

hva planen er for videre utvikling

***

<h2 align = "center"> Refferences  </h2>
