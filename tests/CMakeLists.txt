add_executable(test_add test_add.cpp)
target_link_libraries(test_add PRIVATE simdvec GTest::Main)
add_test(NAME simd.vector.add COMMAND test_add)

add_executable(test_cosine_similarity test_cosine_similarity.cpp)
target_link_libraries(test_cosine_similarity PRIVATE simdvec GTest::Main)
add_test(NAME simd.vector.cosine_similarity COMMAND test_cosine_similarity)

add_executable(test_dot_product test_dot_product.cpp)
target_link_libraries(test_dot_product PRIVATE simdvec GTest::Main)
add_test(NAME simd.vector.dot_product COMMAND test_dot_product)

add_executable(test_l1 test_l1.cpp)
target_link_libraries(test_l1 PRIVATE simdvec GTest::Main)
add_test(NAME simd.vector.l1 COMMAND test_l1)

add_executable(test_l2 test_l2.cpp)
target_link_libraries(test_l2 PRIVATE simdvec GTest::Main)
add_test(NAME simd.vector.l2 COMMAND test_l2)

add_executable(test_sub test_sub.cpp)
target_link_libraries(test_sub PRIVATE simdvec GTest::Main)
add_test(NAME simd.vector.sub COMMAND test_sub)
