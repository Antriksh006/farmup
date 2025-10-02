package com.farmup.farmup_java_backend.controller;

import com.farmup.farmup_java_backend.config.JwtTokenProvider;
import com.farmup.farmup_java_backend.model.User;
import com.farmup.farmup_java_backend.repository.UserRepository;
import com.farmup.farmup_java_backend.service.OtpService;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;


@RestController
@RequestMapping("auth/user")
public class AuthController {

    private final OtpService otpService;
    private final UserRepository userRepository;
    private final JwtTokenProvider jwtTokenProvider;

    public AuthController(OtpService otpService, UserRepository userRepository, JwtTokenProvider jwtTokenProvider) {
        this.otpService = otpService;
        this.userRepository = userRepository;
        this.jwtTokenProvider = jwtTokenProvider;
    }
    // Step 1: Register + Send OTP
    @PostMapping("/register")
    public String register(@RequestParam String name, @RequestParam String phone) {
        User user = userRepository.findByPhone(phone).orElse(new User(name, phone));
        user.setVerified(false); // reset verification for fresh OTP
        userRepository.save(user);

        otpService.sendOtp(phone);
        return "OTP sent to " + phone;
    }

    // Step 2: Verify OTP + Issue JWT
    @PostMapping("/verify")
    public String verify(@RequestParam String phone, @RequestParam String otp) {
        if (otpService.verifyOtp(phone, otp)) {
            User user = userRepository.findByPhone(phone).orElse(null);
            if (user != null) {
                user.setVerified(true);
                userRepository.save(user);

                UserDetails userDetails = org.springframework.security.core.userdetails.User
                        .withUsername(user.getPhone())
                        .authorities("ROLE_USER")
                        .build();

                String token = jwtTokenProvider.generateToken(userDetails);
                return "✅ Login successful. JWT: " + token;
            }
        }
        return "❌ Invalid OTP or User not found";
    }
}
