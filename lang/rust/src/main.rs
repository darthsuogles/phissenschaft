fn main() {
    println!("salut tout le monde!");
    let v = vec![1, 2, 3, 4, 5]; // v: Vec<i32>
    for elem in &v {
        println!("{}", elem)
    }
}
