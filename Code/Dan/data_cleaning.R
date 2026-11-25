library(dplyr)
library(stringr)

# Load raw data

data_train <- read.csv('../data_raw/prom_300_all/train.csv')
data_dev <- read.csv('../data_raw/prom_300_all/dev.csv')
data_test <- read.csv('../data_raw/prom_300_all/test.csv')

# Drop any training rows that leaked into dev and test

# This drops 39 rows
data_dev <- data_dev |> anti_join(data_train, by = 'sequence')

# This drops 20 rows
data_test <- data_test |> anti_join(data_train, by = 'sequence')

# Drop any test rows that leaked into dev

# This drops 1 row
data_dev <- data_dev |> anti_join(data_test, by = 'sequence')

# Look for duplicate sequences
#   If found and they have the same label, collapse to 1 copy.
#   If found and they have different labels, drop all copies.

data_all <- rbind(data_train |> mutate(set = 'train'),
                  data_dev |> mutate(set = 'dev'),
                  data_test |> mutate(set = 'test'))

data_all |> count(set)

#     set     n
# 1   dev  5880
# 2  test  5901
# 3 train 47356

# Drop all copies of each duplicate where the label is inconsistent across copies
data_all <- data_all |>
  group_by(sequence, set) |>
  mutate(n_copies = n(),
         consistent_label = (n_distinct(label) == 1)) |> # TRUE if 1 consistent label
  filter(consistent_label == TRUE) |>
  ungroup()

data_all |> count(set)

#     set     n
# 1   dev  5880
# 2  test  5901
# 3 train 47334

# Dropped 22 rows in train (not sure how many sequences this is)

# Collapse copies (which at this point will have consistent labels) to 1 copy.
data_all <- data_all |>
  distinct(sequence, label, set)

data_all |>
  count(set) |>
  mutate(pct = n / sum(n) * 100)

#     set     n    pct
# 1   dev  5879   9.963
# 2  test  5898   9.995 
# 3 train 47231  80.042

# Remove sequences containing any letters not in {'A', 'T', 'C', 'G'}... which we determined to be just one sequence with 'N'

data_all <- data_all |>
  filter(!str_detect(sequence, 'N'))

data_all |>
  count(set) |>
  mutate(pct = n / sum(n) * 100)

#     set     n    pct
# 1   dev  5878   9.963
# 2  test  5898   9.995 
# 3 train 47231  80.042

# Removed nothing

data_train_clean <- data_all |> filter(set == 'train') |> select(-set)
data_test_clean <- data_all |> filter(set == 'test') |> select(-set)
data_dev_clean <- data_all |> filter(set == 'dev') |> select(-set)

write.csv(data_train_clean, '../data/prom_300_all/train.csv', row.names = FALSE)
write.csv(data_test_clean, '../data/prom_300_all/test.csv', row.names = FALSE)
write.csv(data_dev_clean, '../data/prom_300_all/dev.csv', row.names = FALSE)
