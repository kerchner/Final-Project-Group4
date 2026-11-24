# Cleaned data

Raw data can be found in `../data_raw`  

Data has been cleaned using `Dan/data_cleaning.R` to do the following:  

- Remove any sequences from test and from dev that also appear in train.  This removed 39 rows from dev and 20 rows from test.
- Remove any sequences from dev that also appear in test.  This removed 1 row from dev.
- Look for duplicate sequences within each set.
   - If found and they have different labels, drop all copies. This dropped 22 rows, all in train.  
   - If found and they have the same label, collapse to 1 copy. This dropped 1 row in dev, 3 rows in test, and 103 rows in train.

The cleaning rules slightly favored the training set, but the ratio of training to test and dev remained 80.0/10.0/10.0 even after cleaning:

```
Train 47231 (80.0%)
Test   5898 (10.0%)
Dev    5879 (10.0%)

Total 59008
```
