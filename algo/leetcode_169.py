
def majorityElementSort(nums):
    return sorted(nums)[len(nums) // 2]

def majorityElementCounter(nums):
    from collections import Counter
    return Counter(nums).most_common(1)[0][0]

# A number is selected as @major only if
# the suffix after it is last selected
# has more of its type than any other
# Assuming the @major is not selected,
# then we can remove the suffix of @nums.
# The element @major consist half of the prefix.
# Apply the same argument on this prefix,
# until we have an array of element one,
# which must be [@major]. 
def majorityElement(nums):
    cnt = 1
    major = nums[0]
    for a in nums[1:]:
        if a == major:
            cnt += 1
        elif 0 == cnt:
            major = a; cnt = 1;
        else:
            cnt -= 1
    return major
