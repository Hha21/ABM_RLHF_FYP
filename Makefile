# COMPILER, FLAGS
CXX = g++
CXXFLAGS = -O3 -std=c++17

# DIRECTORIES
SRC_DIR = src_cpp
BUILD_DIR = build
BIN_DIR = bin

SRC_COMMON = $(wildcard $(SRC_DIR)/*.cpp)

OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC_COMMON))

TARGET = $(BIN_DIR)/env

default: $(TARGET)

bpe: $(TARGET)

$(TARGET): $(OBJS)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ 


$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

run: $(TARGET)
	./$(TARGET)