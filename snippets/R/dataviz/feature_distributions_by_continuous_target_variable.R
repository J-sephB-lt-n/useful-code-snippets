#
# TAGS: bivariate|dataviz|dependence|distribution|distributions|plot|regression|relationship|target|variable|visualise|x|y
# DESCRIPTION: Separately visualise the relationship between target variable Y and each feature X

library(tidyverse) # install.packages("tidyverse")
library(ggridges) # install.packages("ggridges")
library(ggExtra) # install.packages("ggExtra")
library(grid) # install.packages("grid")

data <- tibble(
  y = rnorm(100), # Continuous target variable
  x1 = factor(sample(letters[1:3], 100, replace = TRUE)), # Categorical variable with 3 levels
  x2 = runif(100), # Continuous variable
  x3 = factor(sample(letters[4:6], 100, replace = TRUE)), # Categorical variable with 3 levels
  x4 = rnorm(100) # Continuous variable
)

categorical_feature_varnames <- data %>%
  select(where(is.factor)) %>%
  names()
continuous_feature_varnames <- data %>%
  select(where(is.numeric)) %>%
  names()

# For categorical variables, show distribution of Y within each level
for (varname in categorical_feature_varnames) {
  print(
    data %>%
      ggplot(
        .,
        aes(
          x = y,
          y = .data[[varname]],
          fill = factor(stat(quantile))
        )
      ) +
      stat_density_ridges(
        geom = "density_ridges_gradient",
        calc_ecdf = TRUE,
        quantiles = 4,
        quantile_lines = TRUE,
        color = "black" # line colour
      ) +
      scale_fill_viridis_d(name = "Quartiles")
  )
  # grid.newpage() # this is required if knitting from RMarkdown (stops graphs plotted on top of one another)
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
    # grid.newpage() # this is required if knitting from RMarkdown (stops graphs plotted on top of one another)
  )
}
