CSC173 Association Rule Mining Project Proposal

Student: Kervin Lemuel Paalisbo, 2020-0076
Date: December 12, 2025

1. Project Title

Analyzing Associations Between Crime Rates and Socioeconomic Factors Using Association Rule Mining

2. Problem Statement
  It's very important to know the underlying causes of crime to be able to understand why the crimes are happening in the first place.
  Understanding how different socioeconomic conditions relate to the crime rate is a longtime challenge for urban planning and community development.
  Traditional statistical models often capture only linear relationships, overlooking multidimensional patterns. This final project proposes using Association
  Rule Mining to uncover hidden relationships between poverty levels, unemployment, education and crime. The findings may help provide insights relevant to
  social policy and crime prevention within local communities.

3. Objectives
   Since the dataset I chose is not suited for Association Rule Mining(not transaction style and not binary data like yes or no),  I need to transform the dataset first
   Implement Association Rule Mining
   Discover meaningful associations between socioenomic conditions and crime
   Present results using metrics.

4. Dataset Plan
   Source: UCI Machine Learning Repository – Communities and Crime Dataset (≈1,994 rows, 147 attributes)

   Variables of Interest (example classes/categories):
   Poverty Level (Low / Medium / High)
   Crime Rate (Low / Medium / High)
   Education Level (Low / High)
   Unemployment Rate (Low / High)
   
   Acquisition: Public dataset downloaded directly from UCI.

5. Technical Approach
  Architecture Sketch

  Association Rule Mining workflow:
  
  Data cleaning & missing value handling
  
  Discretization of continuous variables
  
  Transaction encoding
  
  ARM algorithm (Apriori or FP-Growth)
  
  Rule evaluation and visualization
  
  Techniques / Tools
  
  Model: Apriori
  
  Frameworks: Python (pandas, mlxtend)
  
  Environment: Google Colab
6. Expected Challenges & Mitigations
  Challenge: Dataset contains mostly continuous values
  Solution: Apply binning/discretization (equal-width, equal-frequency, or domain-informed).
  
  Challenge: High dimensionality (147 attributes)
  Solution: Perform feature selection or correlation filtering to reduce noise.
  
  Challenge: Sparse or redundant rules
  Solution: Use metrics like lift and conviction to filter for meaningful rules.
