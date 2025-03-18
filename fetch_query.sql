SELECT *
FROM your_table
ORDER BY your_column
OFFSET 20 ROWS FETCH NEXT 10 ROWS ONLY;
 This skips first 20 rows, then gets the next 10 rows → Rows 21 to 30.
