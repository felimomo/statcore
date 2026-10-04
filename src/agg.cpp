template <typename T>

T welf_mean (T curr_mean, T new_val, int curr_size) 
{
    return (curr_size * curr_mean + new_val) / (curr_size + 1);
}