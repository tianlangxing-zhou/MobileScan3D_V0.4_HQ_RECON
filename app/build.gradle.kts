import java.util.Properties

val localProperties = Properties().apply {
    val file = rootProject.file("local.properties")
    if (file.exists()) {
        file.inputStream().use { load(it) }
    }
}

fun requiredLocalPath(name: String): String {
    return localProperties.getProperty(name)
        ?.replace("\\", "/")
        ?: error(
            "Missing '$name' in local.properties. " +
                "Please configure Ceres/OpenCV paths."
        )
}

fun optionalLocalPath(name: String): String? {
    return localProperties.getProperty(name)?.replace("\\", "/")
}

val ceresSourceDir = requiredLocalPath("ceres.sourceDir")
val ceresBuildDir = requiredLocalPath("ceres.buildDir")
// OpenCV 4.12 已 vendoring 进仓库（third_party/opencv4android 头 + jniLibs 运行库），
// opencv.sdkDir 仅作为可选覆盖；不设则 CMake 使用仓库内默认，链接库==运行库保证 4.12 一致。
val opencvSdkDir: String? = optionalLocalPath("opencv.sdkDir")

fun gitOutput(vararg args: String): String {
    return try {
        val process = ProcessBuilder("git", *args)
            .directory(rootProject.projectDir)
            .redirectErrorStream(true)
            .start()
        val output = process.inputStream.bufferedReader().use { it.readText().trim() }
        if (process.waitFor() == 0) output else "unknown"
    } catch (_: Exception) {
        "unknown"
    }
}

val gitSha = gitOutput("rev-parse", "--short=8", "HEAD")
val gitDirty = gitOutput("status", "--porcelain").let { it != "unknown" && it.isNotBlank() }
val gitBuildId = if (gitDirty) "$gitSha-dirty" else gitSha
val buildTimestamp = System.currentTimeMillis()

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.mobilescan3d"
    compileSdk = 36
    ndkVersion = "27.1.12297006"

    buildFeatures {
        buildConfig = true
    }

    defaultConfig {
        applicationId = "com.mobilescan3d"
        minSdk = 26
        targetSdk = 36
        versionCode = 167
        versionName = "0.13.27-review"
        buildConfigField("String", "GIT_COMMIT", "\"$gitBuildId\"")
        buildConfigField("long", "BUILD_TIME_MS", "${buildTimestamp}L")
        ndk { abiFilters += listOf("arm64-v8a") }
        externalNativeBuild {
            cmake {
                cppFlags += listOf(
                    "-std=c++20",
                    "-O3"
                )

                arguments += listOf(
                    "-DCERES_SOURCE_DIR=$ceresSourceDir",
                    "-DCERES_BUILD_DIR=$ceresBuildDir"
                )
                if (opencvSdkDir != null) {
                    arguments += "-DOPENCV_ANDROID_SDK=$opencvSdkDir"
                }
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildTypes {
        release { isMinifyEnabled = false }
        debug { isDebuggable = true }
    }

    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt") } }
    packaging { jniLibs.useLegacyPackaging = false }
    androidResources { noCompress += "tflite" }
}

dependencies {
    implementation("androidx.core:core-ktx:1.17.0")
    implementation("androidx.activity:activity-ktx:1.11.0")
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
}
