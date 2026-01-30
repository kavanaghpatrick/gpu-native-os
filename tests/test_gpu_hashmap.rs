//! Tests for GPU-native HashMap using Cuckoo hashing
//!
//! THE GPU IS THE COMPUTER.

use gpu_std::collections::HashMap;

#[test]
fn test_hashmap_new() {
    let map: HashMap<i32, i32> = HashMap::new();
    assert_eq!(map.len(), 0);
    assert!(map.is_empty());
}

#[test]
fn test_hashmap_insert_get() {
    let mut map = HashMap::new();

    // Insert some values
    assert_eq!(map.insert(1, 100), None);
    assert_eq!(map.insert(2, 200), None);
    assert_eq!(map.insert(3, 300), None);

    // Verify they exist
    assert_eq!(map.get(&1), Some(&100));
    assert_eq!(map.get(&2), Some(&200));
    assert_eq!(map.get(&3), Some(&300));
    assert_eq!(map.get(&4), None);

    assert_eq!(map.len(), 3);
}

#[test]
fn test_hashmap_update() {
    let mut map = HashMap::new();

    map.insert(1, 100);
    assert_eq!(map.get(&1), Some(&100));

    // Update existing key
    let old = map.insert(1, 999);
    assert_eq!(old, Some(100));
    assert_eq!(map.get(&1), Some(&999));

    // Length should not change
    assert_eq!(map.len(), 1);
}

#[test]
fn test_hashmap_remove() {
    let mut map = HashMap::new();

    map.insert(1, 100);
    map.insert(2, 200);
    map.insert(3, 300);

    assert_eq!(map.len(), 3);

    // Remove middle element
    let removed = map.remove(&2);
    assert_eq!(removed, Some(200));
    assert_eq!(map.len(), 2);

    // Verify it's gone
    assert_eq!(map.get(&2), None);

    // Other elements still exist
    assert_eq!(map.get(&1), Some(&100));
    assert_eq!(map.get(&3), Some(&300));

    // Remove non-existent key
    assert_eq!(map.remove(&999), None);
}

#[test]
fn test_hashmap_contains_key() {
    let mut map = HashMap::new();

    map.insert(42, "hello");

    assert!(map.contains_key(&42));
    assert!(!map.contains_key(&99));
}

#[test]
fn test_hashmap_clear() {
    let mut map = HashMap::new();

    for i in 0..100 {
        map.insert(i, i * 10);
    }

    assert_eq!(map.len(), 100);

    map.clear();

    assert_eq!(map.len(), 0);
    assert!(map.is_empty());
    assert_eq!(map.get(&50), None);
}

#[test]
fn test_hashmap_many_entries() {
    let mut map = HashMap::new();

    // Insert many entries to test eviction
    for i in 0..500 {
        map.insert(i, i * 2);
    }

    assert_eq!(map.len(), 500);

    // Verify all entries
    for i in 0..500 {
        assert_eq!(map.get(&i), Some(&(i * 2)), "Missing key {}", i);
    }
}

#[test]
fn test_hashmap_get_mut() {
    let mut map = HashMap::new();

    map.insert(1, 100);

    // Modify in place
    if let Some(value) = map.get_mut(&1) {
        *value = 999;
    }

    assert_eq!(map.get(&1), Some(&999));
}

#[test]
fn test_hashmap_entry_or_insert() {
    let mut map = HashMap::new();

    // Insert via entry API
    *map.entry(1).or_insert(100) = 100;
    assert_eq!(map.get(&1), Some(&100));

    // Update via entry API
    *map.entry(1).or_insert(999) = 200;
    assert_eq!(map.get(&1), Some(&200));
}

#[test]
fn test_hashmap_iter() {
    let mut map = HashMap::new();

    map.insert(1, 10);
    map.insert(2, 20);
    map.insert(3, 30);

    let pairs = map.iter();
    assert_eq!(pairs.len(), 3);

    // Check all pairs are present (order not guaranteed)
    let mut found = [false; 3];
    for (k, v) in pairs {
        match k {
            1 => { assert_eq!(v, 10); found[0] = true; }
            2 => { assert_eq!(v, 20); found[1] = true; }
            3 => { assert_eq!(v, 30); found[2] = true; }
            _ => panic!("Unexpected key: {}", k),
        }
    }
    assert!(found.iter().all(|&x| x), "Not all pairs found");
}

#[test]
fn test_hashmap_keys_values() {
    let mut map = HashMap::new();

    map.insert(1, 10);
    map.insert(2, 20);
    map.insert(3, 30);

    let keys = map.keys();
    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&1));
    assert!(keys.contains(&2));
    assert!(keys.contains(&3));

    let values = map.values();
    assert_eq!(values.len(), 3);
    assert!(values.contains(&10));
    assert!(values.contains(&20));
    assert!(values.contains(&30));
}

#[test]
fn test_hashmap_negative_keys() {
    let mut map = HashMap::new();

    map.insert(-1, "negative one");
    map.insert(0, "zero");
    map.insert(1, "positive one");

    assert_eq!(map.get(&-1), Some(&"negative one"));
    assert_eq!(map.get(&0), Some(&"zero"));
    assert_eq!(map.get(&1), Some(&"positive one"));
}

#[test]
fn test_hashmap_string_values() {
    let mut map: HashMap<i32, &str> = HashMap::new();

    map.insert(1, "hello");
    map.insert(2, "world");

    assert_eq!(map.get(&1), Some(&"hello"));
    assert_eq!(map.get(&2), Some(&"world"));
}

#[test]
fn test_hashmap_collision_handling() {
    let mut map = HashMap::new();

    // Insert many entries with potentially colliding hashes
    // The Cuckoo eviction should handle this
    for i in 0..1000 {
        map.insert(i * 64, i);  // Multiples of 64 may hash similarly
    }

    // Verify all entries
    for i in 0..1000 {
        assert_eq!(map.get(&(i * 64)), Some(&i), "Missing key {}", i * 64);
    }
}

#[test]
fn test_hashmap_capacity() {
    let map: HashMap<i32, i32> = HashMap::with_capacity(100);

    // Capacity should be at least what we asked for
    assert!(map.capacity() >= 100);
}
