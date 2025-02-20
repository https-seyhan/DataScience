WITH ranked_rows AS (
    SELECT 
        id, 
        column1, 
        column2, 
        created_at,  -- Change this to your timestamp column
        ROW_NUMBER() OVER (PARTITION BY id ORDER BY created_at DESC) AS rn
    FROM your_table
)
SELECT id, column1, column2, created_at
FROM ranked_rows
WHERE rn = 1;

SELECT t.*
FROM your_table t
JOIN (
    SELECT id, MAX(created_at) AS max_created_at
    FROM your_table
    GROUP BY id
) sub ON t.id = sub.id AND t.created_at = sub.max_created_at;
