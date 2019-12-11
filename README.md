# Semester-Project
Code for my Semester Thesis 2018 for the Master's degree in Robotics, Systems and Control at ETH Zurich.

Abstract:
The focus of this thesis lies on the low voltage distribution grid, which is of primary
concern due to the integration of future technologies. Therefore, long-term predictions
until 2035 are obtained for an aggregation of around 60 consumers. This requires to
consider different scenarios for the development of the electricity consumption and the
adaptation of technologies such as photovoltaic cells, heat pumps and electric vehicles.
Furthermore, smart meter data from the city of Basel was available to analyze consumer
behaviour. It is proposed to extract underlying patterns by K-Means clustering and to
use a semi-parametric model relying solely on external variables. The model is fitted
by linear regression and a neural network which are compared in terms of accuracy and
speed. A characteristic of the proposed model is its dependence on temperature and
irradiation which is exploited to obtain a density forecast for the electricity demand.
Temperature and irradiation simulations are performed by a seasonal bootstrapping
method to obtain enough variation in the samples for an appropriate estimation of
the underlying distribution. The forecasts are used to identify key times defined by
the most profound changes of the consumer behaviour. This knowledge at hand, a
discussion is provided for the overloading of low voltage distribution lines and distribution
transformers. Finally, measures are proposed to circumvent the revealed issues.
