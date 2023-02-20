# Ask the user for at least 4 numbers
print("Enter at least 4 numbers, separated by spaces:")
numbers = list(map(int, input().split()))

# Keep asking for numbers until there are at least 4
while len(numbers) < 4:
  print("Please enter at least 4 numbers")
  numbers = list(map(int, input().split()))

# Try to arrange the numbers in a matrix with the smallest possible difference between the number of rows and columns
success = False
for i in range(2, len(numbers)):
  for j in range(2, len(numbers)):
    if i * j >= len(numbers):
      matrix = [numbers[k:k+j] for k in range(0, len(numbers), j)]
      print("Matrix with", i, "rows and", j, "columns:")
      print("\n".join(" ".join(str(cell) for cell in row) for row in matrix))
      success = True
      break
  if success:
    break

# If it's not possible to create a matrix, print an error message
if not success:
  print("Error: Cannot create a matrix with the given numbers")
