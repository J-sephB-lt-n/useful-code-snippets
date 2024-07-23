#
# TAGS: bivariate|dataviz|dependence|distribution|distributions|plot|regression|relationship|target|variable|visualise|x|y
# DESCRIPTION: Separately visualise the relationship between target variable Y and each feature X

library(tidyverse) # install.packages("tidyverse")
library(ggridges) # install.packages("ggridges")
library(ggExtra) # install.packages("ggExtra")

data <- tibble(
  y = rnorm(100), # Continuous target variable
  x1 = factor(sample(letters[1:3], 100, replace = TRUE)), # Categorical variable with 3 levels
  x2 = runif(100), # Continuous variable
  x3 = factor(sample(letters[4:6], 100, replace = TRUE)), # Categorical variable with 3 levels
  x4 = rnorm(100) # Continuous variable
)

categorical_feature_varnames <- c("x1", "x3")
continuous_feature_varnames <- c("x2", "x4")

# For categorical variables, show distribution of Y within each level
for (varname in categorical_feature_varnames) {
  print(
    data %>%
      ggplot(., aes(x = y, y = .data[[varname]])) +
      geom_density_ridges(
        jittered_points = TRUE, position = "raincloud",
        alpha = 0.7, scale = 0.9
      )
  )
}

# for continuous variables, show scatterplot with marginal histograms
for (varname in continuous_feature_varnames) {
  print(
    ggMarginal(
      ggplot(data, aes(x = .data[[varname]], y = y)) +
        geom_point(alpha = 0.7),
      type = "histogram",
      col = "black",
      fill = "orange"
    )
  )
}
