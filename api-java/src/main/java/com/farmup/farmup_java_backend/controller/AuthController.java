package com.farmup.farmup_java_backend.controller;

import com.farmup.farmup_java_backend.config.JwtTokenProvider;
import com.farmup.farmup_java_backend.model.User;
import com.farmup.farmup_java_backend.repository.UserRepository;
import com.farmup.farmup_java_backend.service.EmailService;
import com.farmup.farmup_java_backend.service.OtpService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;


@RestController
@RequestMapping("auth/user")
public class AuthController {

    private final HashMap<String, String> otpStore = new HashMap<>();

    private final OtpService otpService;
    private final UserRepository userRepository;
    private final JwtTokenProvider jwtTokenProvider;
    private final EmailService emailService;

    public AuthController(OtpService otpService, UserRepository userRepository, JwtTokenProvider jwtTokenProvider, EmailService emailService) {
        this.otpService = otpService;
        this.userRepository = userRepository;
        this.jwtTokenProvider = jwtTokenProvider;
        this.emailService = emailService;
    }
    // Step 1: Register + Send OTP
    @PostMapping("/register")
    public String register(@RequestParam String name, @RequestParam String phone, @RequestParam String email) {
        User user = userRepository.findByPhone(phone).orElse(new User(name, phone, email));
        user.setVerified(false); // reset verification for fresh OTP
        userRepository.save(user);

        otpService.sendOtp(phone);
        String otp = String.valueOf((int) (Math.random() * 9000) + 1000);
        otpStore.put(email, otp);
        emailService.sendOtpEmail(email,otp );
        return "OTP sent to " + phone;
    }

    // Step 2: Verify OTP + Issue JWT
    @PostMapping("/verify-phone")
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

    @PostMapping("/verify-email")
    public ResponseEntity<?> verifyEmail(@RequestParam String email, @RequestParam String otp) {
        String storedOtp = otpStore.get(email);
        if (storedOtp == null) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("No OTP found for this email or OTP expired");
        }

        if (!storedOtp.equals(otp)) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("Invalid OTP");
        }

        return ResponseEntity.ok().build();
    }
}
