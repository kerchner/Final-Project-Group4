# Cleaned data

Raw data can be found in `../data_raw`  

Data has been cleaned using `Dan/data_cleaning.R` to do the following:  

- Remove any sequences from test and from dev that also appear in train.  This removed 39 rows from dev and 20 rows from test.
- Remove any sequences from dev that also appear in test.  This removed 1 row from dev.
- Look for duplicate sequences within each set.
   - If found and they have different labels, drop all copies. This dropped 22 rows, all in train.  
   - If found and they have the same label, collapse to 1 copy. This dropped 1 row in dev, 3 rows in test, and 103 rows in train.
- Remove any sequence containing [IUPAC nucleotide codes](https://www.bioinformatics.org/sms/iupac.html) other than `A`, `C`, `T`, `G`. This dropped 1 row containing an `N` in dev.

The cleaning rules slightly favored the training set, but the ratio of training to test and dev remained 80.0/10.0/10.0 even after cleaning:

|       | Before cleaning | After cleaning   | Rows Removed |
|-------|------------------|------------------|--------------|
| Train | 47356 (80.0%)    | 47231 (80.04%)   | 125 (0.3%)   |
| Test  | 5920 (10.0%)     | 5898 (9.99%)     | 22 (0.4%)    |
| Dev   | 5920 (10.0%)     | 5878 (9.96%)     | 42 (0.7%)    |
| Total | 59196            | 59007            | 189 (0.3%)   |
